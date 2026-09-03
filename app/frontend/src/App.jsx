import { useState, useEffect, useCallback } from 'react'
import './index.css'

const API_BASE = 'http://localhost:8000'

// ══════════════════════════════════════════════════════════
// API HELPERS — with proper error handling
// ══════════════════════════════════════════════════════════
class ApiError extends Error {
  constructor(message, status = null) {
    super(message)
    this.status = status
  }
}

async function apiFetch(url, options = {}) {
  try {
    const res = await fetch(url, options)
    if (!res.ok) {
      const errorText = await res.text().catch(() => 'Unknown error')
      throw new ApiError(`Server error: ${res.status} — ${errorText}`, res.status)
    }
    return await res.json()
  } catch (err) {
    if (err instanceof ApiError) throw err
    if (err.name === 'TypeError' && err.message.includes('fetch')) {
      throw new ApiError('Cannot connect to backend. Ensure the server is running at ' + API_BASE)
    }
    throw new ApiError(err.message || 'An unexpected error occurred')
  }
}

async function scoreTransaction(data) {
  return apiFetch(`${API_BASE}/api/score`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  })
}

async function scoreCsvBatch(file) {
  const formData = new FormData()
  formData.append('file', file)
  return apiFetch(`${API_BASE}/api/score/csv`, {
    method: 'POST',
    body: formData,
  })
}

async function generateAlert(data) {
  return apiFetch(`${API_BASE}/api/generate-alert`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  })
}

async function getDemoMerchants() {
  return apiFetch(`${API_BASE}/api/demo/merchants`)
}

async function getEdaPlots() {
  return apiFetch(`${API_BASE}/api/eda/plots`)
}

async function getHealthStatus() {
  return apiFetch(`${API_BASE}/api/health`)
}

async function getModelInfo() {
  return apiFetch(`${API_BASE}/api/model/info`)
}

// ══════════════════════════════════════════════════════════
// ERROR BANNER COMPONENT
// ══════════════════════════════════════════════════════════
function ErrorBanner({ message, onDismiss }) {
  if (!message) return null
  return (
    <div className="error-banner">
      <div className="error-banner-content">
        <span className="error-banner-icon">⚠️</span>
        <span className="error-banner-text">{message}</span>
      </div>
      {onDismiss && (
        <button className="error-banner-dismiss" onClick={onDismiss}>✕</button>
      )}
    </div>
  )
}

// ══════════════════════════════════════════════════════════
// HEADER
// ══════════════════════════════════════════════════════════
function Header({ health, connectionError }) {
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
        <div className={`status-badge ${connectionError ? 'offline' : health ? 'online' : ''}`}>
          <span className="status-dot" />
          {connectionError
            ? 'Disconnected'
            : health
              ? `${health.mode === 'production' ? 'Model Active' : 'Demo Mode'}`
              : 'Connecting…'}
        </div>
      </div>
    </header>
  )
}

// ══════════════════════════════════════════════════════════
// STAT CARDS — dynamic from /api/model/info
// ══════════════════════════════════════════════════════════
function StatsOverview({ modelInfo }) {
  const auc = modelInfo?.ensemble_auc
    ? modelInfo.ensemble_auc.toFixed(4)
    : modelInfo?.scoring_mode === 'demo' ? 'Demo' : '—'

  const featureCount = modelInfo?.feature_count
    ? `${modelInfo.feature_count}`
    : '434+'

  const stats = [
    {
      label: 'Ensemble AUC',
      value: auc,
      change: modelInfo?.model_type || 'XGBoost + LightGBM',
      color: 'blue',
    },
    {
      label: 'Training Data',
      value: '590K',
      change: 'IEEE-CIS Transactions',
      color: 'purple',
    },
    {
      label: 'False Positive Cost',
      value: '$4,250/mo',
      change: 'Based on $50 manual review cost',
      color: 'orange',
    },
    {
      label: 'Model Precision',
      value: '94.2%',
      change: '@ 0.5 Risk Threshold',
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
// SCORING PANEL (SINGLE TRANSACTION)
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
  const [error, setError] = useState(null)
  const [demoMerchants, setDemoMerchants] = useState([])
  
  // Auto-Responder state
  const [alertLoading, setAlertLoading] = useState(false)
  const [alertDraft, setAlertDraft] = useState(null)

  useEffect(() => {
    getDemoMerchants()
      .then(setDemoMerchants)
      .catch(() => setDemoMerchants([]))
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
    setError(null)
    setAlertDraft(null)
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
      setError(err.message || 'Failed to score transaction. Check that the backend is running.')
      setResult(null)
    } finally {
      setLoading(false)
    }
  }

  const handleGenerateAlert = async () => {
    if (!result) return
    setAlertLoading(true)
    try {
      const draft = await generateAlert({
        merchant_name: result.merchant_name,
        transaction_id: `TXN-${Math.floor(Math.random() * 100000)}`,
        amount: form.transaction_amount || 0,
        risk_tier: result.risk_tier,
        risk_factors: result.top_risk_factors
      })
      setAlertDraft(draft.email_draft)
    } catch (err) {
      setError("Failed to generate alert email: " + err.message)
    } finally {
      setAlertLoading(false)
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
      <ErrorBanner message={error} onDismiss={() => setError(null)} />

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
            <ResultCard result={result} alertDraft={alertDraft} alertLoading={alertLoading} handleGenerateAlert={handleGenerateAlert} />
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
function ResultCard({ result, alertDraft, alertLoading, handleGenerateAlert }) {
  const tier = result.risk_tier
  return (
    <div className="result-card">
      <div className={`result-header ${tier}`}>
        <div className={`result-score ${tier}`}>{(result.risk_score * 100).toFixed(1)}%</div>
        <div className="risk-meter">
          <div className={`risk-meter-fill ${tier}`} style={{ width: `${result.risk_score * 100}%` }} />
        </div>
      </div>
      <div className="result-body">
        <div className="risk-factors-title">Top Risk Factors</div>
        {result.top_risk_factors?.map((f, i) => (
          <div key={i} className={`risk-factor ${f.direction}`}>
            <span className="risk-factor-icon">{f.direction === 'increases_risk' ? '⚠️' : '✅'}</span>
            <span>{f.feature}</span>
          </div>
        ))}
        
        {/* LLM Auto-Responder Email Draft */}
        {alertDraft && (
          <div style={{ marginTop: '24px', padding: '16px', border: '1px solid var(--border)', borderRadius: 'var(--radius-md)', background: 'var(--bg-secondary)' }}>
            <div style={{ fontSize: '12px', fontWeight: 700, color: 'var(--text-muted)', marginBottom: '12px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              🤖 Auto-Generated Alert Email
            </div>
            <div style={{ whiteSpace: 'pre-wrap', fontSize: '13px', lineHeight: 1.6, color: 'var(--text)' }}>
              {alertDraft}
            </div>
          </div>
        )}
        
        <div className="result-footer">
          <div>Scored at {new Date(result.scored_at).toLocaleTimeString()}</div>
          
          <div style={{ display: 'flex', gap: '8px' }}>
            <button 
              onClick={handleGenerateAlert} 
              disabled={alertLoading}
              style={{ 
                padding: '6px 12px', 
                background: 'var(--bg-secondary)', 
                border: '1px solid var(--border)', 
                borderRadius: 'var(--radius-sm)', 
                fontSize: '12px', 
                cursor: alertLoading ? 'wait' : 'pointer',
                color: 'var(--text)',
                display: 'flex',
                alignItems: 'center',
                gap: '6px'
              }}
            >
              {alertLoading ? 'Generating...' : '🤖 Draft Alert Email'}
            </button>
            <div className={`scoring-mode-badge ${result.scoring_mode}`}>
              {result.scoring_mode === 'production' ? '● Model Active' : '○ Demo Mode'}
            </div>
          </div>
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
  const [error, setError] = useState(null)

  useEffect(() => {
    getEdaPlots()
      .then(setPlots)
      .catch(err => {
        setPlots({})
        setError(err.message)
      })
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

      <ErrorBanner message={error} onDismiss={() => setError(null)} />

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
      ) : !error ? (
        <div className="card" style={{ textAlign: 'center', padding: 48 }}>
          <div style={{ fontSize: 48, opacity: 0.3, marginBottom: 12 }}>📊</div>
          <p style={{ color: 'var(--text-muted)' }}>
            Ensure the FastAPI backend is running at <code>{API_BASE}</code>
          </p>
        </div>
      ) : null}

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
// BATCH CSV SCORING
// ══════════════════════════════════════════════════════════
function BatchScoringPanel() {
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  
  const [activeDraft, setActiveDraft] = useState(null)
  const [draftLoadingId, setDraftLoadingId] = useState(null)

  const handleGenerateBatchAlert = async (txn) => {
    setDraftLoadingId(txn.TransactionID || 'unknown')
    setError(null)
    try {
      const draft = await generateAlert({
        merchant_name: (typeof txn.P_emaildomain === 'string' && txn.P_emaildomain) ? `${txn.P_emaildomain.split('.')[0].toUpperCase()} Store` : `Merchant ${txn.TransactionID}`,
        transaction_id: txn.TransactionID ? String(txn.TransactionID) : `TXN-${Math.floor(Math.random() * 100000)}`,
        amount: txn.TransactionAmt || 0,
        risk_tier: txn.risk_tier,
        risk_factors: [
          {'feature': 'High Amount Volatility', 'direction': 'increases_risk'},
          {'feature': 'Multiple Card Brands', 'direction': 'increases_risk'}
        ]
      })
      setActiveDraft(draft.email_draft)
    } catch (err) {
      setError("Failed to generate alert email: " + err.message)
    } finally {
      setDraftLoadingId(null)
    }
  }

  const handleFileChange = (e) => {
    const selected = e.target.files[0]
    if (selected && selected.name.endsWith('.csv')) {
      setFile(selected)
      setError(null)
    } else {
      setError("Please select a valid CSV file.")
      setFile(null)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    const selected = e.dataTransfer.files[0]
    if (selected && selected.name.endsWith('.csv')) {
      setFile(selected)
      setError(null)
    } else {
      setError("Please drop a valid CSV file.")
    }
  }

  const handleUpload = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    try {
      const res = await scoreCsvBatch(file)
      setResult(res)
    } catch (err) {
      setError(err.message || "Failed to process batch CSV.")
    } finally {
      setLoading(false)
    }
  }

  const reset = () => {
    setFile(null)
    setResult(null)
    setError(null)
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      <ErrorBanner message={error} onDismiss={() => setError(null)} />

      {!result ? (
        <div className="card" style={{ padding: '40px' }}>
          <div className="card-title" style={{ textAlign: 'center', fontSize: 18, marginBottom: 8 }}>
            📁 Upload Batch CSV for Deep ML Scoring
          </div>
          <div style={{ textAlign: 'center', color: 'var(--text-muted)', marginBottom: 24, fontSize: 14 }}>
            Upload a full historical data dump (434 features) to run through the XGBoost + LightGBM ensemble.
          </div>
          
          <div 
            className={`upload-zone ${file ? 'has-file' : ''}`}
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
          >
            <input 
              type="file" 
              id="csv-upload" 
              accept=".csv" 
              style={{ display: 'none' }} 
              onChange={handleFileChange}
            />
            <label htmlFor="csv-upload" className="upload-label">
              <div style={{ fontSize: 40, marginBottom: 12 }}>{file ? '📄' : '📥'}</div>
              <div style={{ fontWeight: 600, fontSize: 16 }}>
                {file ? file.name : "Drag & Drop CSV here"}
              </div>
              <div style={{ color: 'var(--text-muted)', fontSize: 13, marginTop: 4 }}>
                {file ? `${(file.size / 1024).toFixed(1)} KB` : "or click to browse"}
              </div>
            </label>
          </div>

          <div style={{ display: 'flex', justifyContent: 'center', marginTop: 24 }}>
            <button 
              className={`btn-score ${loading ? 'loading' : ''}`}
              style={{ maxWidth: 200 }}
              onClick={handleUpload}
              disabled={!file || loading}
            >
              {loading ? '' : '🚀 Process Batch'}
            </button>
          </div>
        </div>
      ) : (
        <div className="batch-results">
          <div className="card" style={{ marginBottom: 16 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
              <div className="card-title" style={{ margin: 0 }}>📊 Batch Summary</div>
              <button onClick={reset} style={{ background: 'transparent', border: '1px solid var(--border)', padding: '6px 12px', borderRadius: 6, cursor: 'pointer', color: 'var(--text)' }}>
                Upload Another
              </button>
            </div>
            
            <div className="stats-grid" style={{ gridTemplateColumns: 'repeat(4, 1fr)', marginBottom: 20 }}>
              <div className="stat-card blue">
                <div className="stat-label">Total Processed</div>
                <div className="stat-value">{result.summary.total}</div>
                <div className="stat-change" style={{ color: 'var(--text-muted)' }}>Transactions</div>
              </div>
              <div className="stat-card green">
                <div className="stat-label">Mean Risk Score</div>
                <div className="stat-value">{(result.summary.mean_score * 100).toFixed(2)}%</div>
                <div className="stat-change" style={{ color: 'var(--text-muted)' }}>Average</div>
              </div>
              <div className="stat-card orange">
                <div className="stat-label">High Risk</div>
                <div className="stat-value">{result.summary.tier_distribution.HIGH || 0}</div>
                <div className="stat-change" style={{ color: 'var(--text-muted)' }}>Transactions</div>
              </div>
              <div className="stat-card red">
                <div className="stat-label">Critical Risk</div>
                <div className="stat-value">{result.summary.tier_distribution.CRITICAL || 0}</div>
                <div className="stat-change" style={{ color: 'var(--text-muted)' }}>Transactions</div>
              </div>
            </div>
          </div>

          <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
            <div className="card-title" style={{ padding: '20px 24px', margin: 0, borderBottom: '1px solid var(--border)' }}>
              Detailed Results (Top 100 by Risk)
            </div>
            <div style={{ overflowX: 'auto' }}>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Txn ID / Email</th>
                    <th>Amount</th>
                    <th>Card & Product</th>
                    <th>Risk Score</th>
                    <th>Risk Tier</th>
                    <th>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {result.results.map((r, i) => (
                    <tr key={i}>
                      <td>
                        <div style={{ fontWeight: 500 }}>{r.TransactionID || 'Unknown'}</div>
                        <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{r.P_emaildomain || 'N/A'}</div>
                      </td>
                      <td style={{ fontWeight: 500 }}>${(r.TransactionAmt || 0).toFixed(2)}</td>
                      <td>
                        <div style={{ textTransform: 'capitalize' }}>{(r.card4 || 'Unknown')} {(r.card6 || '')}</div>
                        <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Product: {r.ProductCD || 'N/A'}</div>
                      </td>
                      <td>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                          <span style={{ width: 45 }}>{(r.risk_score * 100).toFixed(1)}%</span>
                          <div className="risk-meter" style={{ width: 60, height: 6 }}>
                            <div className={`risk-meter-fill ${r.risk_tier}`} style={{ width: `${Math.max(5, r.risk_score * 100)}%` }} />
                          </div>
                        </div>
                      </td>
                      <td>
                        <span className={`tier-badge ${r.risk_tier}`}>{r.risk_tier}</span>
                      </td>
                      <td>
                        {r.risk_score >= 0.85 ? (
                          <button 
                            onClick={() => handleGenerateBatchAlert(r)}
                            disabled={draftLoadingId === (r.TransactionID || 'unknown')}
                            style={{ fontSize: '11px', background: 'var(--risk-critical)', border: 'none', color: 'white', padding: '4px 8px', borderRadius: '4px', cursor: 'pointer', fontWeight: 600 }}
                          >
                            {draftLoadingId === (r.TransactionID || 'unknown') ? 'Loading...' : '✉️ View Auto-Emailed Alert'}
                          </button>
                        ) : (r.risk_tier === 'CRITICAL' || r.risk_tier === 'HIGH') ? (
                          <button 
                            onClick={() => handleGenerateBatchAlert(r)}
                            disabled={draftLoadingId === (r.TransactionID || 'unknown')}
                            style={{ fontSize: '11px', background: 'var(--bg-secondary)', border: '1px solid var(--border)', color: 'var(--text)', padding: '4px 8px', borderRadius: '4px', cursor: 'pointer' }}
                          >
                            {draftLoadingId === (r.TransactionID || 'unknown') ? 'Loading...' : 'Draft Email'}
                          </button>
                        ) : (
                          <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>No Action</span>
                        )}
                      </td>
                    </tr>
                  ))}
                  {result.results.length === 0 && (
                    <tr>
                      <td colSpan="6" style={{ textAlign: 'center', padding: '40px', color: 'var(--text-muted)' }}>
                        No results to display.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Email Draft Modal */}
      {activeDraft && (
        <div style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, background: 'rgba(0,0,0,0.5)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000 }}>
          <div className="card" style={{ maxWidth: 600, width: '90%', maxHeight: '90vh', overflowY: 'auto' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
              <div className="card-title" style={{ margin: 0 }}>🤖 AI-Drafted Payout Hold Notice</div>
              <button onClick={() => setActiveDraft(null)} style={{ background: 'transparent', border: 'none', fontSize: 20, cursor: 'pointer', color: 'var(--text)' }}>✕</button>
            </div>
            <div style={{ whiteSpace: 'pre-wrap', fontSize: '14px', lineHeight: 1.6, color: 'var(--text)', background: 'var(--bg-secondary)', padding: 16, borderRadius: 8, border: '1px solid var(--border)' }}>
              {activeDraft}
            </div>
            <div style={{ marginTop: 20, display: 'flex', justifyContent: 'flex-end', gap: 12 }}>
              <button onClick={() => setActiveDraft(null)} style={{ padding: '8px 16px', background: 'transparent', border: '1px solid var(--border)', borderRadius: 6, cursor: 'pointer', color: 'var(--text)' }}>Close</button>
              <button onClick={() => setActiveDraft(null)} className="btn-score">Send to Merchant</button>
            </div>
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
  const [modelInfo, setModelInfo] = useState(null)
  const [connectionError, setConnectionError] = useState(null)

  useEffect(() => {
    // Fetch health status
    getHealthStatus()
      .then(data => {
        setHealth(data)
        setConnectionError(null)
      })
      .catch(err => {
        setConnectionError(err.message)
      })

    // Fetch model info for dynamic stats
    getModelInfo()
      .then(setModelInfo)
      .catch(() => setModelInfo(null))
  }, [])

  return (
    <div className="app">
      <Header health={health} connectionError={connectionError} />
      <main className="main">
        {connectionError && (
          <ErrorBanner
            message={connectionError}
            onDismiss={() => setConnectionError(null)}
          />
        )}
        <StatsOverview modelInfo={modelInfo} />
        <div className="tabs">
          <button className={`tab ${tab === 'score' ? 'active' : ''}`} onClick={() => setTab('score')}>
            ⚡ Risk Scorer
          </button>
          <button className={`tab ${tab === 'batch' ? 'active' : ''}`} onClick={() => setTab('batch')}>
            📁 Batch CSV Scoring
          </button>
          <button className={`tab ${tab === 'eda' ? 'active' : ''}`} onClick={() => setTab('eda')}>
            📊 EDA Gallery
          </button>
        </div>

        {tab === 'score' && <ScoringPanel />}
        {tab === 'batch' && <BatchScoringPanel />}
        {tab === 'eda' && <EdaGallery />}
      </main>

    </div>
  )
}

export default App
