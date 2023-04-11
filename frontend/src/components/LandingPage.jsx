const LandingPage = () => {
  return (
    <div className="flex pt-52">
      <div class="m-8 relative space-y-4">
        <h1 className="text-4xl break-all pb-12 text-gray-600 font-arimo">
          Don't be a slave to the metronome, let Sound Sprint be your new home
        </h1>
        <button className=" pt-4 py-4 px-8 text-lg rounded-full bg-gradient-to-tl from-pink-200 to-indigo-600 opacity-70 hover:bg-gradient-to-br from-yellow-300 to-indigo-600 hover:opacity-80  border border-grey-900 font-arimo">Generate music</button>
      </div>
      <div class="relative w-full max-w-lg">
        <div class="absolute top-0 -left-4 w-72 h-72 bg-purple-300 rounded-full mix-blend-multiply filter blur-xl opacity-70 animate-blob"></div>
        <div class="absolute top-0 -right-4 w-72 h-72 bg-yellow-300 rounded-full mix-blend-multiply filter blur-xl opacity-70 animate-blob animation-delay-2000"></div>
        <div class="absolute -bottom-8 left-20 w-72 h-72 bg-pink-300 rounded-full mix-blend-multiply filter blur-xl opacity-70 animate-blob animation-delay-4000"></div>
        <div class="m-8 relative space-y-4"></div>
      </div>
    </div>
  );
};

export default LandingPage;

