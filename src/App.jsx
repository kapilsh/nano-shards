import { useStore } from './store/store.js'
import ConfigPanel from './components/ConfigPanel.jsx'
import LayerFlow from './components/LayerFlow.jsx'
import MeshView from './components/MeshView.jsx'
import MemoryView from './components/MemoryView.jsx'
import CommsView from './components/CommsView.jsx'

const VIEWS = [
  { key: 'flow', label: 'Layer flow' },
  { key: 'mesh', label: 'Device mesh' },
  { key: 'memory', label: 'Memory' },
  { key: 'comms', label: 'Communication' },
]

export default function App() {
  const { view, setView } = useStore()
  return (
    <div className="app">
      <header className="top">
        <div className="brand">
          <span className="logo">nano<span className="g">-shards</span></span>
          <span className="tagline">exact tensors, on every GPU</span>
        </div>
        <nav className="tabs">
          {VIEWS.map((v) => (
            <button key={v.key} type="button" className={view === v.key ? 'on' : ''} onClick={() => setView(v.key)}>
              {v.label}
            </button>
          ))}
        </nav>
      </header>
      <main className="body">
        <ConfigPanel />
        <div className="stage">
          {view === 'flow' && <LayerFlow />}
          {view === 'mesh' && <MeshView />}
          {view === 'memory' && <MemoryView />}
          {view === 'comms' && <CommsView />}
        </div>
      </main>
    </div>
  )
}
