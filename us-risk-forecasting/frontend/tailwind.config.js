/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: '#0f62fe',
        secondary: '#393939',
        success: '#24a148',
        warning: '#f1c21b',
        danger: '#da1e28',
        background: '#f4f4f4',
        card: '#ffffff',
      },
      fontFamily: {
        sans: ['Inter', 'IBM Plex Sans', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
