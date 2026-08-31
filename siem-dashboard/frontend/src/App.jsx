import React from 'react'
import { Dashboard } from './components/Dashboard'

function App() {
  return (
    <div className="min-h-screen bg-background text-white selection:bg-primary/30">
      <div className="fixed inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-primary/10 via-background to-background -z-10" />
      <Dashboard />
    </div>
  )
}

export default App
