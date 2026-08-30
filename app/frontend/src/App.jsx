import { useState, useEffect, useCallback } from 'react'
import './index.css'

const API_BASE = 'http://localhost:8000'

// ══════════════════════════════════════════════════════════
// API HELPERS
// ══════════════════════════════════════════════════════════
async function scoreTransaction(data) {
  const res = await fetch(`${API_BASE}/api/score`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  })
  return res.json()
}

async function getDemoMerchants() {
  const res = await fetch(`${API_BASE}/api/demo/merchants`)
  return res.json()
}

async function getEdaPlots() {
  const res = await fetch(`${API_BASE}/api/eda/plots`)
  return res.json()
}

async function getHealthStatus() {
  const res = await fetch(`${API_BASE}/api/health`)
  return res.json()
}

// ══════════════════════════════════════════════════════════
// HEADER
// ══════════════════════════════════════════════════════════
function Header({ health }) {
  return (
    <header className="header">
      <div className="header-left">
        <div className="header-logo">🛡️</div>
        <div>
          <div className="header-title">Merchant Risk Engine</div>
          <div className="header-subtitle">Post-Onboarding Fraud Detection</div>
        </div>
      </div>
      <div className="header-right">
        <div className={`status-badge ${health ? 'online' : ''}`}>
          <span className="status-dot" />
          {health ? `${health.mode === 'production' ? 'Model Active' : 'Demo Mode'}` : 'Connecting…'}
        </div>
      </div>
    </header>
  )
}

// ══════════════════════════════════════════════════════════
// STAT CARDS
// ══════════════════════════════════════════════════════════
function StatsOverview({ demoResults }) {
  const stats = [
    {
      label: 'Ensemble AUC',
      value: '0.9567',
      change: 'XGBoost + LightGBM',
      color: 'blue',
    },
    {
      label: 'Training Data',
      value: '590K',
      change: 'IEEE-CIS Transactions',
      color: 'purple',
    },
    {
      label: 'Fraud Rate',
      value: '3.50%',
      change: '28:1 Class Imbalance',
      color: 'red',
    },
    {
      label: 'Features',
      value: '475+',
      change: '47 engineered features',
      color: 'green',
    },
  ]

  return (
    <div className="stats-grid">
      {stats.map((s, i) => (
        <div key={i} className={`stat-card ${s.color}`}>
          <div className="stat-label">{s.label}</div>
          <div className="stat-value">{s.value}</div>
          <div className="stat-change" style={{ color: 'var(--text-muted)' }}>{s.change}</div>
        </div>
      ))}
    </div>
  )
}

// ══════════════════════════════════════════════════════════
// SCORING FORM
// ══════════════════════════════════════════════════════════
const DEFAULT_FORM = {
  merchant_name: '',
  transaction_amount: '',
  product_cd: 'W',
  card_brand: 'visa',
  card_type: 'debit',
  email_domain: 'gmail.com',
  device_type: 'desktop',
  hour_of_day: 12,
  is_international: false,
}

function ScoringPanel() {
  const [form, setForm] = useState(DEFAULT_FORM)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [demoMerchants, setDemoMerchants] = useState([])

  useEffect(() => {
    getDemoMerchants().then(setDemoMerchants).catch(() => setDemoMerchants([]))
  }, [])

  const handleChange = (key, value) => {
    setForm(prev => ({ ...prev, [key]: value }))
  }

  const handleDemoClick = (merchant) => {
    const newForm = { ...DEFAULT_FORM }
    Object.keys(DEFAULT_FORM).forEach(key => {
      if (merchant[key] !== undefined) newForm[key] = merchant[key]
    })
    newForm.merchant_name = merchant.name
    setForm(newForm)
  }

  const handleDemoSelect = (e) => {
    const selectedName = e.target.value;
    if (!selectedName) return;
    const merchant = demoMerchants.find(m => m.name === selectedName);
    if (merchant) {
      handleDemoClick(merchant);
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    try {
      const payload = {
        ...form,
        transaction_amount: parseFloat(form.transaction_amount) || 0,
        hour_of_day: parseInt(form.hour_of_day) || 12,
      }
      const res = await scoreTransaction(payload)
      setResult(res)
    } catch (err) {
      console.error('Scoring failed:', err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
      <div style={{ marginBottom: '0', display: 'flex', alignItems: 'center', gap: '12px', background: 'var(--bg-card)', padding: '10px 16px', borderRadius: 'var(--radius-lg)', border: '1px solid var(--border)' }}>
        <div style={{ fontSize: 14, fontWeight: 700, display: 'flex', alignItems: 'center', gap: 8 }}>
          <span>🎭</span> Quick Test Profile:
        </div>
        <select className="form-select" style={{ flex: 1, maxWidth: '400px' }} onChange={handleDemoSelect} defaultValue="">
          <option value="" disabled>Select a demo profile to autofill...</option>
          {demoMerchants.map((m, i) => (
            <option key={i} value={m.name}>{m.name} — {m.description}</option>
          ))}
        </select>
      </div>

      <div className="scoring-layout">
        <div className="card">
          <div className="card-title">⚡ Score a Transaction</div>
          <form onSubmit={handleSubmit}>
            <div className="form-grid">
              <div className="form-group full-width">
                <label className="form-label">Merchant Name</label>
                <input className="form-input" type="text" placeholder="e.g. Sharma Electronics, Jaipur"
                  value={form.merchant_name}
                  onChange={e => handleChange('merchant_name', e.target.value)} />
              </div>
              <div className="form-group">
                <label className="form-label">Amount (USD)</label>
                <input className="form-input" type="number" step="0.01" placeholder="45.99"
                  value={form.transaction_amount}
                  onChange={e => handleChange('transaction_amount', e.target.value)} />
              </div>
              <div className="form-group">
                <label className="form-label">Product Code</label>
                <select className="form-select" value={form.product_cd}
                  onChange={e => handleChange('product_cd', e.target.value)}>
                  <option value="W">W — Web</option>
                  <option value="H">H — High Value</option>
                  <option value="C">C — Card Present</option>
                  <option value="S">S — Subscription</option>
                  <option value="R">R — Recurring</option>
                </select>
              </div>
              <div className="form-group">
                <label className="form-label">Card Brand</label>
                <select className="form-select" value={form.card_brand}
                  onChange={e => handleChange('card_brand', e.target.value)}>
                  <option value="visa">Visa</option>
                  <option value="mastercard">Mastercard</option>
                  <option value="discover">Discover</option>
                  <option value="american express">American Express</option>
                </select>
              </div>
              <div className="form-group">
                <label className="form-label">Card Type</label>
                <select className="form-select" value={form.card_type}
                  onChange={e => handleChange('card_type', e.target.value)}>
                  <option value="debit">Debit</option>
                  <option value="credit">Credit</option>
                </select>
              </div>
              <div className="form-group">
                <label className="form-label">Email Domain</label>
                <select className="form-select" value={form.email_domain}
                  onChange={e => handleChange('email_domain', e.target.value)}>
                  <option value="gmail.com">gmail.com</option>
                  <option value="yahoo.com">yahoo.com</option>
                  <option value="outlook.com">outlook.com</option>
                  <option value="hotmail.com">hotmail.com</option>
                  <option value="mail.com">mail.com</option>
                  <option value="icloud.com">icloud.com</option>
                  <option value="anonymous.com">anonymous.com</option>
                </select>
              </div>
              <div className="form-group">
                <label className="form-label">Device Type</label>
                <select className="form-select" value={form.device_type}
                  onChange={e => handleChange('device_type', e.target.value)}>
                  <option value="desktop">Desktop</option>
                  <option value="mobile">Mobile</option>
                </select>
              </div>
              <div className="form-group">
                <label className="form-label">Hour of Day (0-23)</label>
                <input className="form-input" type="number" min="0" max="23"
                  value={form.hour_of_day}
                  onChange={e => handleChange('hour_of_day', e.target.value)} />
              </div>
              <div className="form-checkbox-row">
                <input className="form-checkbox" type="checkbox" id="intl"
                  checked={form.is_international}
                  onChange={e => handleChange('is_international', e.target.checked)} />
                <label htmlFor="intl" style={{ fontSize: '13px', fontWeight: 600 }}>International Transaction</label>
              </div>
            </div>
            <button type="submit" className={`btn-score ${loading ? 'loading' : ''}`} disabled={loading}>
              {loading ? '' : '🛡️ Analyse Risk'}
            </button>
          </form>
        </div>

        <div>
          {result ? (
            <ResultCard result={result} />
          ) : (
            <div className="card" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
              <div className="card-title">🎯 Analysis Result</div>
              <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: 250, flexDirection: 'column', gap: 12 }}>
                <div style={{ fontSize: 40, opacity: 0.3 }}>🛡️</div>
                <div style={{ color: 'var(--text-muted)', fontSize: 14, textAlign: 'center' }}>
                  Fill in merchant details and click<br /><strong>Analyse Risk</strong> to score a transaction
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// ══════════════════════════════════════════════════════════
// RESULT CARD
// ══════════════════════════════════════════════════════════
function ResultCard({ result }) {
  const tier = result.risk_tier
  return (
    <div className="result-card">
      <div className={`result-header ${tier}`}>
        <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 8 }}>
          {result.merchant_name || 'Unknown Merchant'}
        </div>
        <div className={`result-score ${tier}`}>{(result.risk_score * 100).toFixed(1)}%</div>
        <div className="risk-meter">
          <div className={`risk-meter-fill ${tier}`} style={{ width: `${result.risk_score * 100}%` }} />
        </div>
        <div className={`result-tier`} style={{ color: `var(--risk-${tier.toLowerCase()})` }}>
          {result.risk_label}
        </div>
        <div className="result-action">{result.risk_action}</div>
      </div>
      <div className="result-body">
        <div className="risk-factors-title">Top Risk Factors</div>
        {result.top_risk_factors?.map((f, i) => (
          <div key={i} className={`risk-factor ${f.direction}`}>
            <span className="risk-factor-icon">{f.direction === 'increases_risk' ? '⚠️' : '✅'}</span>
            <span>{f.feature}</span>
          </div>
        ))}
        <div style={{ marginTop: 16, fontSize: 11, color: 'var(--text-muted)' }}>
          Scored at {new Date(result.scored_at).toLocaleString()}
        </div>
      </div>
    </div>
  )
}



// ══════════════════════════════════════════════════════════
// EDA GALLERY TAB
// ══════════════════════════════════════════════════════════
function EdaGallery() {
  const [plots, setPlots] = useState({})
  const [selectedPlot, setSelectedPlot] = useState(null)

  useEffect(() => {
    getEdaPlots().then(setPlots).catch(() => setPlots({}))
  }, [])

  const formatName = (name) => name.replace(/[_-]/g, ' ').replace('.png', '')

  const allPlots = Object.entries(plots).flatMap(([category, files]) =>
    files.map(f => ({ category, file: f, url: `${API_BASE}/api/eda/plot/${category}/${f}` }))
  ).slice(0, 8)

  return (
    <div>
      <h2 style={{ fontSize: 18, fontWeight: 700, marginBottom: 4 }}>📊 Selected Exploratory Data Analysis</h2>
      <p style={{ fontSize: 13, color: 'var(--text-muted)', marginBottom: 20 }}>
        Validated plots from the IEEE-CIS dataset — all generated on raw data before encoding
      </p>
      
      {allPlots.length > 0 ? (
        <div className="eda-gallery">
          {allPlots.map((p, i) => (
            <div key={i} className="eda-plot-card" onClick={() => setSelectedPlot(p.url)}>
              <div className="eda-plot-image-wrapper">
                <img src={p.url} alt={p.file} loading="lazy" />
              </div>
              <div className="eda-plot-label">{formatName(p.file)}</div>
            </div>
          ))}
        </div>
      ) : (
        <div className="card" style={{ textAlign: 'center', padding: 48 }}>
          <div style={{ fontSize: 48, opacity: 0.3, marginBottom: 12 }}>📊</div>
          <p style={{ color: 'var(--text-muted)' }}>
            Ensure the FastAPI backend is running at <code>{API_BASE}</code>
          </p>
        </div>
      )}

      {selectedPlot && (
        <div className="lightbox-overlay" onClick={() => setSelectedPlot(null)}>
          <div className="lightbox-content" onClick={e => e.stopPropagation()}>
            <button className="lightbox-close" onClick={() => setSelectedPlot(null)}>✕</button>
            <img src={selectedPlot} alt="EDA Plot" className="lightbox-img" />
          </div>
        </div>
      )}
    </div>
  )
}

// ══════════════════════════════════════════════════════════
// APP
// ══════════════════════════════════════════════════════════
function App() {
  const [tab, setTab] = useState('score')
  const [health, setHealth] = useState(null)

  useEffect(() => {
    getHealthStatus().then(setHealth).catch(() => null)
  }, [])

  return (
    <div className="app">
      <Header health={health} />
      <main className="main">
        <StatsOverview />
        <div className="tabs">
          <button className={`tab ${tab === 'score' ? 'active' : ''}`} onClick={() => setTab('score')}>
            ⚡ Risk Scorer
          </button>
          <button className={`tab ${tab === 'eda' ? 'active' : ''}`} onClick={() => setTab('eda')}>
            📊 EDA Gallery
          </button>
        </div>

        {tab === 'score' && <ScoringPanel />}
        {tab === 'eda' && <EdaGallery />}
      </main>

    </div>
  )
}

export default App
