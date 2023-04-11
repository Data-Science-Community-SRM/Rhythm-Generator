/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily:{
        'arimo' : ['Arimo', 'sans-serif']
      },
      keyframes:{
        Animation: {
          blob : "blob 1s infinite"
        },
        blob:{
          "0%":{
            transform: " translate (0px,0px) scale(1)",
          }, 
          "33%":{
            transform: " translate (30px,-50px) scale(1.2)",
          }, 
          "66%":{
            transform: "translate (-20px,20px) scale(0.8)",
          }, 
          "100%":{
            transform: "translate (0px,0px) scale(1)",
          }, 
        }
      }
    },
  },
  plugins: [],
}

