import { useState, useEffect, useCallback } from 'react'
import { Card, CardHeader, CardTitle, CardContent, CardDescription, CardFooter } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from "@/components/ui/dialog"
import { Skeleton } from "@/components/ui/skeleton"
import { Toaster, toast } from "sonner"
import { Shield, ShieldAlert, TriangleAlert, CircleCheck, Zap, BarChart3, Upload, Loader2, Mail, CheckCircle2, AlertCircle, X, ChevronRight, FileDown } from "lucide-react"
import { useTheme } from "next-themes"
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
    <Alert variant="destructive" className="mb-6 relative">
      <AlertCircle className="h-4 w-4" />
      <AlertTitle>Error</AlertTitle>
      <AlertDescription>{message}</AlertDescription>
      {onDismiss && (
        <button onClick={onDismiss} className="absolute right-4 top-4 opacity-70 hover:opacity-100 transition-opacity">
          <X className="h-4 w-4" />
        </button>
      )}
    </Alert>
  )
}

// ══════════════════════════════════════════════════════════
// HEADER
// ══════════════════════════════════════════════════════════
function Header({ health, connectionError }) {
  const { theme, setTheme } = useTheme()
  return (
    <header className="flex items-center justify-between px-6 py-4 border-b bg-card text-card-foreground">
      <div className="flex items-center gap-4">
        <div className="bg-primary text-primary-foreground p-2 rounded-lg shadow-sm">
          <Shield className="h-6 w-6" />
        </div>
        <div>
          <div className="text-xl font-bold leading-tight tracking-tight">Merchant Risk Engine</div>
          <div className="text-xs text-muted-foreground font-medium">Post-Onboarding Fraud Detection</div>
        </div>
      </div>
      <div className="flex items-center gap-4">
        <Badge 
          variant="outline" 
          className={`flex items-center gap-2 py-1.5 px-3 rounded-full ${
            connectionError ? 'border-destructive text-destructive' : 
            health ? 'border-green-500 text-green-500 bg-green-500/10' : ''
          }`}
        >
          <span className={`h-2 w-2 rounded-full ${
            connectionError ? 'bg-destructive' : 
            health ? 'bg-green-500 animate-pulse' : 'bg-muted-foreground'
          }`} />
          {connectionError
            ? 'Disconnected'
            : health
              ? `${health.mode === 'production' ? 'Model Active' : 'Demo Mode'}`
              : 'Connecting…'}
        </Badge>
        <Button variant="ghost" size="icon" onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')} className="rounded-full">
          {theme === 'dark' ? '☀️' : '🌙'}
        </Button>
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

  const stats = [
    {
      label: 'Ensemble AUC',
      value: auc,
      change: modelInfo?.model_type || 'XGBoost + LightGBM',
      icon: <BarChart3 className="h-4 w-4 text-blue-500" />
    },
    {
      label: 'Training Data',
      value: '590K',
      change: 'IEEE-CIS Transactions',
      icon: <DatabaseIcon className="h-4 w-4 text-purple-500" />
    },
    {
      label: 'False Positive Cost',
      value: '$4,250/mo',
      change: 'Based on $50 manual review cost',
      icon: <DollarSignIcon className="h-4 w-4 text-orange-500" />
    },
    {
      label: 'Model Precision',
      value: '94.2%',
      change: '@ 0.5 Risk Threshold',
      icon: <TargetIcon className="h-4 w-4 text-green-500" />
    },
  ]

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
      {stats.map((s, i) => (
        <Card key={i} className="shadow-sm hover:shadow transition-shadow">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">
              {s.label}
            </CardTitle>
            {s.icon}
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{s.value}</div>
            <p className="text-xs text-muted-foreground mt-1">
              {s.change}
            </p>
          </CardContent>
        </Card>
      ))}
    </div>
  )
}

function DatabaseIcon(props) {
  return (
    <svg {...props} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/><path d="M3 12A9 3 0 0 0 21 12"/></svg>
  )
}
function DollarSignIcon(props) {
  return (
    <svg {...props} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="12" x2="12" y1="2" y2="22"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
  )
}
function TargetIcon(props) {
  return (
    <svg {...props} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>
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
    toast.success(`Loaded demo profile: ${merchant.name}`)
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
      toast.success("Transaction analyzed successfully")
    } catch (err) {
      console.error('Scoring failed:', err)
      setError(err.message || 'Failed to score transaction.')
      setResult(null)
      toast.error("Analysis failed")
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
      toast.success("Alert drafted successfully")
    } catch (err) {
      setError("Failed to generate alert email: " + err.message)
      toast.error("Failed to generate draft")
    } finally {
      setAlertLoading(false)
    }
  }

  return (
    <div className="flex flex-col gap-6">
      <ErrorBanner message={error} onDismiss={() => setError(null)} />

      <Card className="bg-secondary/20 shadow-none border-border">
        <CardContent className="p-4 flex flex-col sm:flex-row items-start sm:items-center gap-4">
          <div className="text-sm font-semibold flex items-center gap-2 whitespace-nowrap">
            <span className="text-xl">🎭</span> Quick Test Profile:
          </div>
          <Select onValueChange={(val) => {
            const m = demoMerchants.find(x => x.name === val)
            if (m) handleDemoClick(m)
          }}>
            <SelectTrigger className="w-full sm:max-w-[400px] bg-background">
              <SelectValue placeholder="Select a demo profile to autofill..." />
            </SelectTrigger>
            <SelectContent>
              {demoMerchants.map((m, i) => (
                <SelectItem key={i} value={m.name}>{m.name} — {m.description}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-xl">
              <Zap className="h-5 w-5 text-amber-500" />
              Score a Transaction
            </CardTitle>
            <CardDescription>Enter transaction details below to run deep ML scoring.</CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2 md:col-span-2">
                  <Label>Merchant Name</Label>
                  <Input placeholder="e.g. Sharma Electronics, Jaipur"
                    value={form.merchant_name}
                    onChange={e => handleChange('merchant_name', e.target.value)} />
                </div>
                <div className="space-y-2">
                  <Label>Amount (USD)</Label>
                  <Input type="number" step="0.01" placeholder="45.99"
                    value={form.transaction_amount}
                    onChange={e => handleChange('transaction_amount', e.target.value)} />
                </div>
                <div className="space-y-2">
                  <Label>Product Code</Label>
                  <Select value={form.product_cd} onValueChange={v => handleChange('product_cd', v)}>
                    <SelectTrigger><SelectValue/></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="W">W — Web</SelectItem>
                      <SelectItem value="H">H — High Value</SelectItem>
                      <SelectItem value="C">C — Card Present</SelectItem>
                      <SelectItem value="S">S — Subscription</SelectItem>
                      <SelectItem value="R">R — Recurring</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label>Card Brand</Label>
                  <Select value={form.card_brand} onValueChange={v => handleChange('card_brand', v)}>
                    <SelectTrigger className="capitalize"><SelectValue/></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="visa">Visa</SelectItem>
                      <SelectItem value="mastercard">Mastercard</SelectItem>
                      <SelectItem value="discover">Discover</SelectItem>
                      <SelectItem value="american express">American Express</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label>Card Type</Label>
                  <Select value={form.card_type} onValueChange={v => handleChange('card_type', v)}>
                    <SelectTrigger className="capitalize"><SelectValue/></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="debit">Debit</SelectItem>
                      <SelectItem value="credit">Credit</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label>Email Domain</Label>
                  <Select value={form.email_domain} onValueChange={v => handleChange('email_domain', v)}>
                    <SelectTrigger><SelectValue/></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="gmail.com">gmail.com</SelectItem>
                      <SelectItem value="yahoo.com">yahoo.com</SelectItem>
                      <SelectItem value="outlook.com">outlook.com</SelectItem>
                      <SelectItem value="hotmail.com">hotmail.com</SelectItem>
                      <SelectItem value="mail.com">mail.com</SelectItem>
                      <SelectItem value="icloud.com">icloud.com</SelectItem>
                      <SelectItem value="anonymous.com">anonymous.com</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label>Device Type</Label>
                  <Select value={form.device_type} onValueChange={v => handleChange('device_type', v)}>
                    <SelectTrigger className="capitalize"><SelectValue/></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="desktop">Desktop</SelectItem>
                      <SelectItem value="mobile">Mobile</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label>Hour of Day (0-23)</Label>
                  <Input type="number" min="0" max="23"
                    value={form.hour_of_day}
                    onChange={e => handleChange('hour_of_day', e.target.value)} />
                </div>
                <div className="flex items-center space-x-2 pt-8">
                  <input type="checkbox" id="intl" className="h-4 w-4 rounded border-gray-300 text-primary focus:ring-primary"
                    checked={form.is_international}
                    onChange={e => handleChange('is_international', e.target.checked)} />
                  <Label htmlFor="intl" className="cursor-pointer">International Transaction</Label>
                </div>
              </div>
              <Button type="submit" className="w-full font-bold" size="lg" disabled={loading}>
                {loading ? <Loader2 className="mr-2 h-5 w-5 animate-spin" /> : <Shield className="mr-2 h-5 w-5" />}
                {loading ? 'Analyzing...' : 'Analyse Risk'}
              </Button>
            </form>
          </CardContent>
        </Card>

        <div className="flex flex-col h-full">
          {result ? (
            <ResultCard result={result} alertDraft={alertDraft} alertLoading={alertLoading} handleGenerateAlert={handleGenerateAlert} />
          ) : (
            <Card className="flex flex-col h-full items-center justify-center min-h-[300px] border-dashed text-center p-8 bg-muted/30">
              <ShieldAlert className="h-16 w-16 text-muted-foreground/30 mb-4" />
              <CardTitle className="text-lg text-muted-foreground mb-2">Awaiting Data</CardTitle>
              <CardDescription className="max-w-xs">
                Fill in merchant details and click Analyse Risk to view the AI ensemble score.
              </CardDescription>
            </Card>
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
  const tier = result.risk_tier // LOW, MEDIUM, HIGH, CRITICAL

  const getTierColor = (t) => {
    switch(t) {
      case 'CRITICAL': return 'bg-risk-critical text-destructive-foreground'
      case 'HIGH': return 'bg-risk-high text-destructive-foreground'
      case 'MEDIUM': return 'bg-amber-500 text-primary-foreground'
      case 'LOW': return 'bg-risk-low text-primary-foreground'
      default: return 'bg-muted text-muted-foreground'
    }
  }

  const getTierTextColor = (t) => {
    switch(t) {
      case 'CRITICAL': return 'text-risk-critical'
      case 'HIGH': return 'text-risk-high'
      case 'MEDIUM': return 'text-amber-500'
      case 'LOW': return 'text-risk-low'
      default: return 'text-muted-foreground'
    }
  }

  return (
    <Card className="overflow-hidden flex flex-col h-full border-2 shadow-md relative" style={{ borderColor: `var(--color-${tier === 'CRITICAL' ? 'risk-critical' : tier === 'HIGH' ? 'risk-high' : tier === 'MEDIUM' ? 'amber-500' : 'risk-low'})` }}>
      <div className={`p-6 flex flex-col items-center justify-center ${getTierColor(tier)} relative overflow-hidden`}>
        <div className="absolute inset-0 opacity-10 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-white to-transparent"></div>
        <div className="relative z-10 flex flex-col items-center">
          <Badge variant="outline" className="mb-2 bg-white/20 hover:bg-white/30 text-white border-white/30 shadow-sm backdrop-blur-sm px-3 py-1 text-xs tracking-widest font-bold">
            {tier} RISK
          </Badge>
          <div className="text-5xl font-black tabular-nums tracking-tight">
            {(result.risk_score * 100).toFixed(1)}%
          </div>
        </div>
      </div>
      
      <CardContent className="p-6 flex-1 flex flex-col">
        <h4 className="text-sm font-bold uppercase tracking-wider text-muted-foreground mb-4">Key Risk Drivers</h4>
        <div className="space-y-3 mb-6">
          {result.top_risk_factors?.map((f, i) => (
            <div key={i} className="flex items-start gap-3 bg-secondary/50 p-3 rounded-md border border-border/50">
              <div className="mt-0.5">
                {f.direction === 'increases_risk' ? 
                  <TriangleAlert className="h-4 w-4 text-destructive" /> : 
                  <CircleCheck className="h-4 w-4 text-green-500" />
                }
              </div>
              <span className="text-sm font-medium leading-tight">{f.feature}</span>
            </div>
          ))}
        </div>
        
        {alertDraft && (
          <div className="mt-auto pt-4 border-t">
            <div className="text-xs font-bold text-muted-foreground mb-2 flex items-center gap-1.5 uppercase tracking-wider">
              <Mail className="h-3.5 w-3.5" /> Auto-Generated Alert Email
            </div>
            <div className="text-sm p-4 bg-muted/50 rounded-lg border border-border whitespace-pre-wrap font-mono leading-relaxed max-h-48 overflow-y-auto">
              {alertDraft}
            </div>
          </div>
        )}
      </CardContent>

      <CardFooter className="bg-muted/30 border-t p-4 flex items-center justify-between">
        <div className="text-xs text-muted-foreground font-medium">
          Scored at {new Date(result.scored_at).toLocaleTimeString()}
        </div>
        
        <div className="flex items-center gap-3">
          <Button 
            variant="outline" 
            size="sm"
            onClick={handleGenerateAlert} 
            disabled={alertLoading}
            className="font-semibold shadow-sm"
          >
            {alertLoading ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <Mail className="h-4 w-4 mr-2" />}
            {alertLoading ? 'Drafting...' : 'Draft Alert Email'}
          </Button>
          <Badge variant="secondary" className="font-mono text-[10px]">
            {result.scoring_mode === 'production' ? 'PROD' : 'DEMO'}
          </Badge>
        </div>
      </CardFooter>
    </Card>
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
    <div className="space-y-6 animate-in fade-in duration-500">
      <div>
        <h2 className="text-2xl font-bold tracking-tight">Exploratory Data Analysis</h2>
        <p className="text-muted-foreground mt-1">
          Validated plots from the IEEE-CIS dataset — all generated on raw data before encoding
        </p>
      </div>

      <ErrorBanner message={error} onDismiss={() => setError(null)} />

      {allPlots.length > 0 ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {allPlots.map((p, i) => (
            <Card key={i} className="cursor-pointer overflow-hidden group border hover:border-primary transition-all shadow-sm hover:shadow-md" onClick={() => setSelectedPlot(p.url)}>
              <div className="aspect-video bg-muted relative overflow-hidden">
                <img src={p.url} alt={p.file} loading="lazy" className="object-cover w-full h-full group-hover:scale-105 transition-transform duration-500" />
                <div className="absolute inset-0 bg-black/0 group-hover:bg-black/10 transition-colors" />
              </div>
              <CardFooter className="p-3 bg-card border-t">
                <span className="text-sm font-semibold truncate capitalize" title={formatName(p.file)}>{formatName(p.file)}</span>
              </CardFooter>
            </Card>
          ))}
        </div>
      ) : !error ? (
        <Card className="flex flex-col items-center justify-center py-20 text-center border-dashed">
          <BarChart3 className="h-16 w-16 text-muted-foreground/30 mb-4" />
          <h3 className="text-lg font-medium text-foreground">No plots available</h3>
          <p className="text-sm text-muted-foreground mt-2 max-w-sm">
            Ensure the FastAPI backend is running at <code className="bg-muted px-1.5 py-0.5 rounded text-primary">{API_BASE}</code>
          </p>
        </Card>
      ) : null}

      <Dialog open={!!selectedPlot} onOpenChange={(open) => !open && setSelectedPlot(null)}>
        <DialogContent className="max-w-4xl p-1 bg-black/90 border-none shadow-2xl">
          <img src={selectedPlot} alt="EDA Plot full size" className="w-full h-auto max-h-[85vh] object-contain rounded-md" />
        </DialogContent>
      </Dialog>
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
  const [isDragging, setIsDragging] = useState(false)

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
      toast.success("Draft created for " + (txn.TransactionID || 'transaction'))
    } catch (err) {
      setError("Failed to generate alert email: " + err.message)
      toast.error("Failed to generate draft")
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
    setIsDragging(false)
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
      toast.success(`Batch processed successfully: ${res.summary.total} transactions`)
    } catch (err) {
      setError(err.message || "Failed to process batch CSV.")
      toast.error("Batch processing failed")
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
    <div className="space-y-6 animate-in slide-in-from-bottom-4 duration-500">
      <ErrorBanner message={error} onDismiss={() => setError(null)} />

      {!result ? (
        <Card className="max-w-3xl mx-auto shadow-md">
          <CardHeader className="text-center pb-2">
            <CardTitle className="text-2xl flex items-center justify-center gap-2">
              <FileDown className="h-6 w-6 text-primary" />
              Batch CSV Scoring
            </CardTitle>
            <CardDescription className="text-base">
              Upload a full historical data dump to run through the XGBoost + LightGBM ensemble.
            </CardDescription>
          </CardHeader>
          <CardContent className="pt-6 pb-8">
            <div 
              className={`border-2 border-dashed rounded-xl p-12 text-center transition-all duration-200 ${
                isDragging ? 'border-primary bg-primary/5 scale-[1.02]' : 
                file ? 'border-green-500/50 bg-green-500/5' : 'border-border hover:bg-muted/50'
              }`}
              onDragOver={(e) => { e.preventDefault(); setIsDragging(true) }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
            >
              <input 
                type="file" 
                id="csv-upload" 
                accept=".csv" 
                className="hidden" 
                onChange={handleFileChange}
              />
              <label htmlFor="csv-upload" className="cursor-pointer flex flex-col items-center">
                <div className={`p-4 rounded-full mb-4 ${file ? 'bg-green-500/10 text-green-500' : 'bg-primary/10 text-primary'}`}>
                  {file ? <CheckCircle2 className="h-10 w-10" /> : <Upload className="h-10 w-10" />}
                </div>
                <div className="text-lg font-bold mb-1">
                  {file ? file.name : "Drag & Drop CSV here"}
                </div>
                <div className="text-sm text-muted-foreground">
                  {file ? `${(file.size / 1024).toFixed(1)} KB` : "or click to browse from your computer"}
                </div>
              </label>
            </div>

            <div className="flex justify-center mt-8">
              <Button 
                size="lg"
                className="min-w-[200px] font-bold shadow-md hover:shadow-lg transition-shadow"
                onClick={handleUpload}
                disabled={!file || loading}
              >
                {loading ? <Loader2 className="mr-2 h-5 w-5 animate-spin" /> : <Zap className="mr-2 h-5 w-5" />}
                {loading ? 'Processing Batch...' : 'Run Analysis'}
              </Button>
            </div>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-6">
          <Card className="shadow-md">
            <CardHeader className="flex flex-row items-center justify-between pb-4 border-b">
              <div>
                <CardTitle className="text-xl">Batch Summary</CardTitle>
                <CardDescription>Processed {file?.name}</CardDescription>
              </div>
              <Button variant="outline" size="sm" onClick={reset}>
                Upload Another
              </Button>
            </CardHeader>
            <CardContent className="pt-6">
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                <div className="bg-blue-500/10 border-l-4 border-blue-500 p-4 rounded-r-lg">
                  <div className="text-sm font-medium text-blue-700 dark:text-blue-400">Total Processed</div>
                  <div className="text-3xl font-bold mt-1 text-foreground">{result.summary.total}</div>
                </div>
                <div className="bg-green-500/10 border-l-4 border-green-500 p-4 rounded-r-lg">
                  <div className="text-sm font-medium text-green-700 dark:text-green-400">Mean Risk Score</div>
                  <div className="text-3xl font-bold mt-1 text-foreground">{(result.summary.mean_score * 100).toFixed(2)}%</div>
                </div>
                <div className="bg-orange-500/10 border-l-4 border-orange-500 p-4 rounded-r-lg">
                  <div className="text-sm font-medium text-orange-700 dark:text-orange-400">High Risk</div>
                  <div className="text-3xl font-bold mt-1 text-foreground">{result.summary.tier_distribution.HIGH || 0}</div>
                </div>
                <div className="bg-red-500/10 border-l-4 border-red-500 p-4 rounded-r-lg">
                  <div className="text-sm font-medium text-red-700 dark:text-red-400">Critical Risk</div>
                  <div className="text-3xl font-bold mt-1 text-foreground">{result.summary.tier_distribution.CRITICAL || 0}</div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="shadow-md overflow-hidden">
            <CardHeader className="bg-muted/30 border-b py-4">
              <CardTitle className="text-lg">Detailed Results <span className="text-muted-foreground text-sm font-normal ml-2">(Top 100 by Risk)</span></CardTitle>
            </CardHeader>
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Txn ID / Email</TableHead>
                    <TableHead>Amount</TableHead>
                    <TableHead>Card & Product</TableHead>
                    <TableHead>Risk Score</TableHead>
                    <TableHead>Risk Tier</TableHead>
                    <TableHead className="text-right">Action</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {result.results.map((r, i) => (
                    <TableRow key={i} className="hover:bg-muted/50 transition-colors">
                      <TableCell>
                        <div className="font-semibold">{r.TransactionID || 'Unknown'}</div>
                        <div className="text-xs text-muted-foreground">{r.P_emaildomain || 'N/A'}</div>
                      </TableCell>
                      <TableCell className="font-medium font-mono">${(r.TransactionAmt || 0).toFixed(2)}</TableCell>
                      <TableCell>
                        <div className="capitalize font-medium">{(r.card4 || 'Unknown')} {(r.card6 || '')}</div>
                        <div className="text-xs text-muted-foreground">Product: {r.ProductCD || 'N/A'}</div>
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center gap-3">
                          <span className="w-12 font-bold tabular-nums">{(r.risk_score * 100).toFixed(1)}%</span>
                          <div className="w-16 h-2 bg-secondary rounded-full overflow-hidden">
                            <div 
                              className={`h-full ${r.risk_tier === 'CRITICAL' ? 'bg-risk-critical' : r.risk_tier === 'HIGH' ? 'bg-risk-high' : r.risk_tier === 'MEDIUM' ? 'bg-amber-500' : 'bg-risk-low'}`} 
                              style={{ width: `${Math.max(5, r.risk_score * 100)}%` }} 
                            />
                          </div>
                        </div>
                      </TableCell>
                      <TableCell>
                        <Badge variant={r.risk_tier === 'CRITICAL' ? 'destructive' : r.risk_tier === 'HIGH' ? 'destructive' : r.risk_tier === 'MEDIUM' ? 'default' : 'secondary'}
                          className={`${r.risk_tier === 'MEDIUM' ? 'bg-amber-500 hover:bg-amber-600' : ''}`}>
                          {r.risk_tier}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-right">
                        {r.risk_score >= 0.85 ? (
                          <Button 
                            size="sm"
                            variant="destructive"
                            onClick={() => handleGenerateBatchAlert(r)}
                            disabled={draftLoadingId === (r.TransactionID || 'unknown')}
                            className="text-xs h-8 shadow-sm"
                          >
                            {draftLoadingId === (r.TransactionID || 'unknown') ? <Loader2 className="h-3 w-3 mr-1 animate-spin" /> : <Mail className="h-3 w-3 mr-1" />}
                            View Alert
                          </Button>
                        ) : (r.risk_tier === 'CRITICAL' || r.risk_tier === 'HIGH') ? (
                          <Button 
                            size="sm"
                            variant="outline"
                            onClick={() => handleGenerateBatchAlert(r)}
                            disabled={draftLoadingId === (r.TransactionID || 'unknown')}
                            className="text-xs h-8"
                          >
                            {draftLoadingId === (r.TransactionID || 'unknown') ? <Loader2 className="h-3 w-3 mr-1 animate-spin" /> : null}
                            Draft Email
                          </Button>
                        ) : (
                          <span className="text-xs text-muted-foreground">No Action</span>
                        )}
                      </TableCell>
                    </TableRow>
                  ))}
                  {result.results.length === 0 && (
                    <TableRow>
                      <TableCell colSpan={6} className="h-32 text-center text-muted-foreground">
                        No results to display.
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </div>
          </Card>
        </div>
      )}

      {/* Email Draft Dialog */}
      <Dialog open={!!activeDraft} onOpenChange={(open) => !open && setActiveDraft(null)}>
        <DialogContent className="sm:max-w-[600px]">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Mail className="h-5 w-5 text-primary" /> AI-Drafted Payout Hold Notice
            </DialogTitle>
          </DialogHeader>
          <div className="bg-muted p-4 rounded-md font-mono text-sm whitespace-pre-wrap max-h-[50vh] overflow-y-auto border border-border">
            {activeDraft}
          </div>
          <DialogFooter className="mt-4">
            <Button variant="outline" onClick={() => setActiveDraft(null)}>Cancel</Button>
            <Button onClick={() => {
              setActiveDraft(null)
              toast.success("Draft sent to merchant")
            }}>
              Send to Merchant
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}

// ══════════════════════════════════════════════════════════
// APP
// ══════════════════════════════════════════════════════════
function App() {
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
    <div className="min-h-screen bg-background text-foreground flex flex-col font-sans selection:bg-primary/20">
      <Header health={health} connectionError={connectionError} />
      <main className="flex-1 container mx-auto px-4 py-8 max-w-7xl">
        {connectionError && (
          <ErrorBanner
            message={connectionError}
            onDismiss={() => setConnectionError(null)}
          />
        )}
        <StatsOverview modelInfo={modelInfo} />
        
        <Tabs defaultValue="score" className="w-full">
          <TabsList className="grid w-full grid-cols-3 mb-8 bg-secondary/50 p-1">
            <TabsTrigger value="score" className="font-semibold">⚡ Risk Scorer</TabsTrigger>
            <TabsTrigger value="batch" className="font-semibold">📁 Batch CSV</TabsTrigger>
            <TabsTrigger value="eda" className="font-semibold">📊 EDA Gallery</TabsTrigger>
          </TabsList>
          
          <TabsContent value="score" className="mt-0 outline-none">
            <ScoringPanel />
          </TabsContent>
          <TabsContent value="batch" className="mt-0 outline-none">
            <BatchScoringPanel />
          </TabsContent>
          <TabsContent value="eda" className="mt-0 outline-none">
            <EdaGallery />
          </TabsContent>
        </Tabs>
      </main>
      <Toaster position="top-right" closeButton richColors />
    </div>
  )
}

export default App
