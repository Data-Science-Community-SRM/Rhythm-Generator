import tensorflow as tf
import numpy as np
import miditoolkit
import pickle
import utils
import time
import os
import tensorflow as tf

def embedding_lookup(lookup_table, x):
    return tf.compat.v1.nn.embedding_lookup(lookup_table, x)


def normal_embedding_lookup(x, n_token, d_embed, d_proj, initializer,
                            proj_initializer, scope='normal_embed', **kwargs):
    emb_scale = d_proj ** 0.5
    with tf.compat.v1.variable_scope(scope):
        lookup_table = tf.compat.v1.get_variable('lookup_table', [n_token, d_embed], initializer=initializer)
        y = embedding_lookup(lookup_table, x)
        if d_proj != d_embed:
            proj_W = tf.compat.v1.get_variable('proj_W', [d_embed, d_proj], initializer=proj_initializer)
            y = tf.einsum('ibe,ed->ibd', y, proj_W)
        else:
            proj_W = None
        ret_params = [lookup_table, proj_W]
    y *= emb_scale
    return y, ret_params


def normal_softmax(hidden, target, n_token, params, scope='normal_softmax', **kwargs):
    def _logit(x, W, b, proj):
        y = x
        if proj is not None:
            y = tf.einsum('ibd,ed->ibe', y, proj)
        return tf.einsum('ibd,nd->ibn', y, W) + b

    params_W, params_projs = params[0], params[1]

    with tf.compat.v1.variable_scope(scope):
        softmax_b = tf.compat.v1.get_variable('bias', [n_token], initializer=tf.zeros_initializer())
        output = _logit(hidden, params_W, softmax_b, params_projs)
        nll = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=output)
    return nll, output


def positional_embedding(pos_seq, inv_freq, bsz=None):
    sinusoid_inp = tf.einsum('i,j->ij', pos_seq, inv_freq)
    pos_emb = tf.concat([tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)], -1)
    if bsz is not None:
        return tf.tile(pos_emb[:, None, :], [1, bsz, 1])
    else:
        return pos_emb[:, None, :]


def positionwise_FF(inp, d_model, d_inner, dropout, kernel_initializer,
                    scope='ff', is_training=True):
    output = inp
    with tf.compat.v1.variable_scope(scope):
        output = tf.keras.layers.Dense(d_inner, activation=tf.nn.relu, 
                                       kernel_initializer=kernel_initializer, name='layer_1')(inp)
        output = tf.keras.layers.Dropout(dropout, name='drop_1')(output, training=is_training)
        output = tf.keras.layers.Dense(d_model, activation=tf.nn.relu, 
                                       kernel_initializer=kernel_initializer, name='layer_2')(output)
        output = tf.keras.layers.Dropout(dropout, name='drop_2')(output, training=is_training)
        output = tf.keras.layers.LayerNormalization(axis=-1)(output + inp)
    return output


def _create_mask(qlen, mlen, same_length=False):
    attn_mask = tf.ones([qlen, qlen])
    mask_u = tf.linalg.band_part(attn_mask, 0, -1)
    mask_dia = tf.linalg.band_part(attn_mask, 0, 0)
    attn_mask_pad = tf.zeros([qlen, mlen])
    ret = tf.concat([attn_mask_pad, mask_u - mask_dia], 1)
    if same_length:
        mask_l = tf.matrix_band_part(attn_mask, -1, 0)
        ret = tf.concat([ret[:, :qlen] + mask_l - mask_dia, ret[:, qlen:]], 1)
    return ret


def _cache_mem(curr_out, prev_mem, mem_len=None):
    if mem_len is None or prev_mem is None:
        new_mem = curr_out
    elif mem_len == 0:
        return prev_mem
    else:
        new_mem = tf.concat([prev_mem, curr_out], 0)[-mem_len:]
    return tf.stop_gradient(new_mem)


def rel_shift(x):
    x_size = tf.shape(x)
    x = tf.pad(x, [[0, 0], [1, 0], [0, 0], [0, 0]])
    x = tf.reshape(x, [x_size[1] + 1, x_size[0], x_size[2], x_size[3]])
    x = tf.slice(x, [1, 0, 0, 0], [-1, -1, -1, -1])
    x = tf.reshape(x, x_size)
    return x


def rel_multihead_attn(w, r, r_w_bias, r_r_bias, attn_mask, mems, d_model,
                       n_head, d_head, dropout, dropatt, is_training,
                       kernel_initializer, scope='rel_attn'):
    scale = 1 / (d_head ** 0.5)
    with tf.compat.v1.variable_scope(scope):
        qlen = tf.shape(w)[0]
        rlen = tf.shape(r)[0]
        bsz = tf.shape(w)[1]

        cat = tf.concat([mems, w], 0) if mems is not None and mems.shape.ndims > 1 else w

        w_heads = tf.keras.layers.Dense(3 * n_head * d_head, use_bias=False, 
                                        kernel_initializer=kernel_initializer, name='qkv')(cat)
        r_head_k = tf.keras.layers.Dense(n_head * d_head, use_bias=False,
                                         kernel_initializer=kernel_initializer, name='r')(r)
        
        w_head_q, w_head_k, w_head_v = tf.split(w_heads, 3, -1)
        w_head_q = w_head_q[-qlen:]

        klen = tf.shape(w_head_k)[0]

        w_head_q = tf.reshape(w_head_q, [qlen, bsz, n_head, d_head])
        w_head_k = tf.reshape(w_head_k, [klen, bsz, n_head, d_head])
        w_head_v = tf.reshape(w_head_v, [klen, bsz, n_head, d_head])

        r_head_k = tf.reshape(r_head_k, [rlen, n_head, d_head])

        rw_head_q = w_head_q + r_w_bias
        rr_head_q = w_head_q + r_r_bias

        AC = tf.einsum('ibnd,jbnd->ijbn', rw_head_q, w_head_k)
        BD = tf.einsum('ibnd,jnd->ijbn', rr_head_q, r_head_k)
        BD = rel_shift(BD)

        attn_score = (AC + BD) * scale
        attn_mask_t = attn_mask[:, :, None, None]
        attn_score = attn_score * (1 - attn_mask_t) - 1e30 * attn_mask_t

        attn_prob = tf.nn.softmax(attn_score, 1)
        attn_prob = tf.keras.layers.Dropout(dropatt)(attn_prob, training=is_training)

        attn_vec = tf.einsum('ijbn,jbnd->ibnd', attn_prob, w_head_v)
        size_t = tf.shape(attn_vec)
        attn_vec = tf.reshape(attn_vec, [size_t[0], size_t[1], n_head * d_head])

        attn_out = tf.keras.layers.Dense(d_model, use_bias=False, 
                                         kernel_initializer=kernel_initializer, name='o')(attn_vec)
        attn_out = tf.keras.layers.Dropout(dropout)(attn_out, training=is_training)
        output = tf.keras.layers.LayerNormalization(axis=-1)(attn_out + w)
        return output


def transformer(dec_inp, target, mems, n_token, n_layer, d_model, d_embed,
                n_head, d_head, d_inner, dropout, dropatt,
                initializer, is_training, proj_initializer=None,
                mem_len=None, cutoffs=[], div_val=1, tie_projs=[],
                same_length=False, clamp_len=-1,
                input_perms=None, target_perms=None, head_target=None,
                untie_r=False, proj_same_dim=True,
                scope='transformer'):
    new_mems = []
    with tf.compat.v1.variable_scope(scope):
        if untie_r:
            r_w_bias = tf.compat.v1.get_variable('r_w_bias', [n_layer, n_head, d_head], initializer=initializer)
            r_r_bias = tf.compat.v1.get_variable('r_r_bias', [n_layer, n_head, d_head], initializer=initializer)
        else:
            r_w_bias = tf.compat.v1.get_variable('r_w_bias', [n_head, d_head], initializer=initializer)
            r_r_bias = tf.compat.v1.get_variable('r_r_bias', [n_head, d_head], initializer=initializer)

        qlen = tf.shape(dec_inp)[0]
        mlen = tf.shape(mems[0])[0] if mems is not None else 0
        klen = qlen + mlen

        if proj_initializer is None:
            proj_initializer = initializer

        embeddings, shared_params = normal_embedding_lookup(
            x=dec_inp,
            n_token=n_token,
            d_embed=d_embed,
            d_proj=d_model,
            initializer=initializer,
            proj_initializer=proj_initializer)
        
        attn_mask = _create_mask(qlen, mlen, same_length)
        
        pos_seq = tf.range(klen - 1, -1, -1.0)
        if clamp_len > 0:
            pos_seq = tf.minimum(pos_seq, clamp_len)
        inv_freq = 1 / (10000 ** (tf.range(0, d_model, 2.0) / d_model))
        pos_emb = positional_embedding(pos_seq, inv_freq)

        output = tf.keras.layers.Dropout(rate=dropout)(embeddings, training=is_training)
        pos_emb = tf.keras.layers.Dropout(rate=dropout)(pos_emb, training=is_training)

        if mems is None:
            mems = [None] * n_layer

        for i in range(n_layer):
            # cache new mems
            new_mems.append(_cache_mem(output, mems[i], mem_len))

            with tf.compat.v1.variable_scope('layer_{}'.format(i)):
                output = rel_multihead_attn(
                    w=output,
                    r=pos_emb,
                    r_w_bias=r_w_bias if not untie_r else r_w_bias[i],
                    r_r_bias=r_r_bias if not untie_r else r_r_bias[i],
                    attn_mask=attn_mask,
                    mems=mems[i],
                    d_model=d_model,
                    n_head=n_head,
                    d_head=d_head,
                    dropout=dropout,
                    dropatt=dropatt,
                    is_training=is_training,
                    kernel_initializer=initializer)

                output = positionwise_FF(
                    inp=output,
                    d_model=d_model,
                    d_inner=d_inner,
                    dropout=dropout,
                    kernel_initializer=initializer,
                    is_training=is_training)

        output = tf.keras.layers.Dropout(dropout)(output, training=is_training)

        loss, logits = normal_softmax(
            hidden=output,
            target=target,
            n_token=n_token,
            params=shared_params)

        return loss, logits, new_mems

class RythmTransformer(object):

    def __init__(self, checkpoint, is_training=False):
        # load dictionary
        self.dictionary_path = '{}/dictionary.pkl'.format(checkpoint)
        self.event2word, self.word2event = pickle.load(open(self.dictionary_path, 'rb'))
        # model settings
        self.x_len = 512
        self.mem_len = 512
        self.n_layer = 12
        self.d_embed = 512
        self.d_model = 512
        self.dropout = 0.1
        self.n_head = 8
        self.d_head = self.d_model // self.n_head
        self.d_ff = 2048
        self.n_token = len(self.event2word)
        self.learning_rate = 0.0002
        # load model
        self.is_training = is_training
        if self.is_training:
            self.batch_size = 4
        else:
            self.batch_size = 1
        self.checkpoint_path = '{}/model'.format(checkpoint)
        self.load_model()

    def load_model(self):
        # placeholders
        self.x = tf.compat.v1.placeholder(tf.int32, shape=[self.batch_size, None])
        self.y = tf.compat.v1.placeholder(tf.int32, shape=[self.batch_size, None])
        self.mems_i = [tf.compat.v1.placeholder(tf.float32, [self.mem_len, self.batch_size, self.d_model]) for _ in range(self.n_layer)]
        # model
        self.global_step = tf.compat.v1.train.get_or_create_global_step()
        initializer = tf.compat.v1.initializers.random_normal(stddev=0.02, seed=None)
        proj_initializer = tf.compat.v1.initializers.random_normal(stddev=0.01, seed=None)
        with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope()):
            xx = tf.transpose(self.x, [1, 0])
            yy = tf.transpose(self.y, [1, 0])
            loss, self.logits, self.new_mem = transformer(
                dec_inp=xx,
                target=yy,
                mems=self.mems_i,
                n_token=self.n_token,
                n_layer=self.n_layer,
                d_model=self.d_model,
                d_embed=self.d_embed,
                n_head=self.n_head,
                d_head=self.d_head,
                d_inner=self.d_ff,
                dropout=self.dropout,
                dropatt=self.dropout,
                initializer=initializer,
                proj_initializer=proj_initializer,
                is_training=self.is_training,
                mem_len=self.mem_len,
                cutoffs=[],
                div_val=-1,
                tie_projs=[],
                same_length=False,
                clamp_len=-1,
                input_perms=None,
                target_perms=None,
                head_target=None,
                untie_r=False,
                proj_same_dim=True)
        self.avg_loss = tf.reduce_mean(loss)
        # vars
        all_vars = tf.compat.v1.trainable_variables()
        grads = tf.gradients(self.avg_loss, all_vars)
        grads_and_vars = list(zip(grads, all_vars))
        all_trainable_vars = tf.reduce_sum([tf.reduce_prod(v.shape) for v in tf.compat.v1.trainable_variables()])
        # optimizer
        decay_lr = tf.compat.v1.train.cosine_decay(
            self.learning_rate,
            global_step=self.global_step,
            decay_steps=300000,
            alpha=0.004)
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=decay_lr)
        self.train_op = optimizer.apply_gradients(grads_and_vars, self.global_step)
        # saver
        self.saver = tf.compat.v1.train.Saver()
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        config.gpu_options.allow_growth = True
        config.gpu_options.polling_inactive_delay_msecs = 10
        self.sess = tf.compat.v1.Session(config=config)
        self.saver.restore(self.sess, self.checkpoint_path)
         # saver

    def temperature_sampling(self, logits, temperature, topk=5):
        probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
        if topk == 1:
            prediction = np.argmax(probs)
        else:
            sorted_index = np.argsort(probs)[::-1]
            candi_index = sorted_index[:topk]
            candi_probs = [probs[i] for i in candi_index]
            # normalize probs
            candi_probs /= sum(candi_probs)
            # choose by predicted probs
            prediction = np.random.choice(candi_index, size=1, p=candi_probs)[0]
        return prediction

    def extract_events(self, input_path):
        note_items, tempo_items = utils.read_items(input_path)
        note_items = utils.quantize_items(note_items)
        max_time = note_items[-1].end
        if 'chord' in self.checkpoint_path:
            chord_items = utils.extract_chords(note_items)
            items = chord_items + tempo_items + note_items
        else:
            items = tempo_items + note_items
        groups = utils.group_items(items, max_time)
        events = utils.item2event(groups)
        return events

    def generate(self, n_target_bar, temperature, output_path, prompt=None):
        # if prompt, load it. Or, random start
        if prompt:
            events = self.extract_events(prompt)
            words = [[self.event2word['{}_{}'.format(e.name, e.value)] for e in events]]
            words[0].append(self.event2word['Bar_None'])
        else:
            words = []
            for _ in range(self.batch_size):
                ws = [self.event2word['Bar_None']]
                if 'chord' in self.checkpoint_path:
                    tempo_classes = [v for k, v in self.event2word.items() if 'Tempo Class' in k]
                    tempo_values = [v for k, v in self.event2word.items() if 'Tempo Value' in k]
                    chords = [v for k, v in self.event2word.items() if 'Chord' in k]
                    ws.append(self.event2word['Position_1/16'])
                    ws.append(np.random.choice(chords))
                    ws.append(self.event2word['Position_1/16'])
                    ws.append(np.random.choice(tempo_classes))
                    ws.append(np.random.choice(tempo_values))
                else:
                    tempo_classes = [v for k, v in self.event2word.items() if 'Tempo Class' in k]
                    tempo_values = [v for k, v in self.event2word.items() if 'Tempo Value' in k]
                    ws.append(self.event2word['Position_1/16'])
                    ws.append(np.random.choice(tempo_classes))
                    ws.append(np.random.choice(tempo_values))
                words.append(ws)
        # initialize mem
        batch_m = [np.zeros((self.mem_len, self.batch_size, self.d_model), dtype=np.float32) for _ in range(self.n_layer)]
        # generate
        original_length = len(words[0])
        initial_flag = 1
        current_generated_bar = 0
        while current_generated_bar < n_target_bar:
            # input
            if initial_flag:
                temp_x = np.zeros((self.batch_size, original_length))
                for b in range(self.batch_size):
                    for z, t in enumerate(words[b]):
                        temp_x[b][z] = t
                initial_flag = 0
            else:
                temp_x = np.zeros((self.batch_size, 1))
                for b in range(self.batch_size):
                    temp_x[b][0] = words[b][-1]
            # prepare feed dict
            feed_dict = {self.x: temp_x}
            for m, m_np in zip(self.mems_i, batch_m):
                feed_dict[m] = m_np
            # model (prediction)
            _logits, _new_mem = self.sess.run([self.logits, self.new_mem], feed_dict=feed_dict)
            # sampling
            _logit = _logits[-1, 0]
            word = self.temperature_sampling(
                logits=_logit, 
                temperature=temperature)
            words[0].append(word)
            if word == self.event2word['Bar_None']:
                current_generated_bar += 1
            # re-new mem
            batch_m = _new_mem
        # write
        if prompt:
            utils.write_midi(
                words=words[0][original_length:],
                word2event=self.word2event,
                output_path=output_path,
                prompt_path=prompt)
        else:
            utils.write_midi(
                words=words[0],
                word2event=self.word2event,
                output_path=output_path,
                prompt_path=None)
    def prepare_data(self, midi_paths):
        # extract events
        all_events = []
        for path in midi_paths:
            events = self.extract_events(path)
            all_events.append(events)
        # event to word
        all_words = []
        for events in all_events:
            words = []
            for event in events:
                e = '{}_{}'.format(event.name, event.value)
                if e in self.event2word:
                    words.append(self.event2word[e])
                else:
                    # OOV
                    if event.name == 'Note Velocity':
                        # replace with max velocity based on our training data
                        words.append(self.event2word['Note Velocity_21'])
                    else:
                        print('something is wrong! {}'.format(e))
            all_words.append(words)
            print(all_words)
        # to training data
        self.group_size = 5
        segments = []
        for words in all_words:
            pairs = []
            for i in range(0, len(words)-self.x_len-1, self.x_len):
                x = words[i:i+self.x_len]
                y = words[i+1:i+self.x_len+1]
                pairs.append([x, y])
            # abandon the last
            for i in np.arange(0, len(pairs)-self.group_size, self.group_size):
                data = pairs[i:i+self.group_size]
                if len(data) == self.group_size:
                    segments.append(data)
        segments = np.array(segments)
        return segments

    def finetune(self, training_data, output_checkpoint_folder):
        if not os.path.exists(output_checkpoint_folder):
            os.makedirs(output_checkpoint_folder)
        # shuffle
        index = np.arange(len(training_data))
        np.random.shuffle(index)
        training_data = training_data[index]
        num_batches = len(training_data) // self.batch_size
        print(num_batches)
        st = time.time()
        for e in range(200):
            total_loss = []
            for i in range(5):
                segments = training_data[self.batch_size*i:self.batch_size*(i+1)]
                batch_m = [np.zeros((self.mem_len, self.batch_size, self.d_model), dtype=np.float32) for _ in range(self.n_layer)]
                for j in range(self.group_size):
                    batch_x = segments[:, j, 0, :]
                    batch_y = segments[:, j, 1, :]
                    # prepare feed dict
                    feed_dict = {self.x: batch_x, self.y: batch_y}
                    for m, m_np in zip(self.mems_i, batch_m):
                        feed_dict[m] = m_np
                    # run
                    _, gs_, loss_, new_mem_ = self.sess.run([self.train_op, self.global_step, self.avg_loss, self.new_mem], feed_dict=feed_dict)
                    batch_m = new_mem_
                    total_loss.append(loss_)
                    print('e: {}'.format( e))
                    print('i:', i)
                    print('segments.shape:', segments.shape)
                    print('batch_x.shape:', batch_x.shape)
                    print('batch_y.shape:', batch_y.shape)

                    print(' Epoch: {}, Step: {}, Loss: {:.5f}, Time: {:.2f}'.format(e, gs_, loss_, time.time()-st))
            self.saver.save(self.sess, os.path.join(output_checkpoint_folder, 'model-{:03d}-{:.3f}'.format(e, np.mean(total_loss))))
            # stop
            if np.mean(total_loss) <= 0.1:
                break
    def close(self):
        self.sess.close()