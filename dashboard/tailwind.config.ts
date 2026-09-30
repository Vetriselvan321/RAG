import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        bg: {
          primary:  '#0B0F19',
          panel:    '#111827',
          card:     '#1A2234',
          elevated: '#1E2A3B',
        },
        accent: {
          blue:   '#4F8CFF',
          purple: '#7C3AED',
          green:  '#22C55E',
          yellow: '#F59E0B',
          red:    '#EF4444',
        },
        ink: {
          primary:   '#F8FAFC',
          secondary: '#94A3B8',
          muted:     '#64748B',
          dim:       '#374151',
        },
        border: 'rgba(255,255,255,0.08)',
      },
      fontFamily: {
        sans: ['var(--font-inter)', 'Inter', 'system-ui', 'sans-serif'],
      },
      boxShadow: {
        'card':        '0 4px 24px rgba(0,0,0,0.4)',
        'glow-blue':   '0 0 24px rgba(79,140,255,0.25)',
        'glow-purple': '0 0 24px rgba(124,58,237,0.25)',
        'input-focus': '0 0 0 2px rgba(79,140,255,0.35), 0 0 16px rgba(79,140,255,0.12)',
      },
      backgroundImage: {
        'gradient-brand':  'linear-gradient(135deg,#4F8CFF 0%,#7C3AED 100%)',
        'gradient-subtle': 'linear-gradient(180deg,rgba(79,140,255,0.08) 0%,transparent 100%)',
      },
      keyframes: {
        'fade-in': {
          '0%':   { opacity: '0', transform: 'translateY(8px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        'slide-in': {
          '0%':   { opacity: '0', transform: 'translateX(-8px)' },
          '100%': { opacity: '1', transform: 'translateX(0)' },
        },
        shimmer: {
          '0%':   { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition:  '200% 0' },
        },
        'typing-dot': {
          '0%,80%,100%': { opacity: '0', transform: 'scale(0.7)' },
          '40%':          { opacity: '1', transform: 'scale(1)'   },
        },
        pulse: {
          '0%,100%': { opacity: '1' },
          '50%':     { opacity: '0.4' },
        },
      },
      animation: {
        'fade-in':    'fade-in 0.3s ease-out forwards',
        'slide-in':   'slide-in 0.25s ease-out forwards',
        'shimmer':    'shimmer 1.6s ease-in-out infinite',
        'typing-dot': 'typing-dot 1.3s ease-in-out infinite',
        'pulse-slow': 'pulse 2.5s ease-in-out infinite',
      },
    },
  },
  plugins: [],
}

export default config
