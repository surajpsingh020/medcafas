/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './app/**/*.{ts,tsx}',
    './components/**/*.{ts,tsx}',
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      colors: {
        risk: {
          low:     '#16a34a',
          caution: '#d97706',
          high:    '#dc2626',
        },
      },
    },
  },
  plugins: [],
};
