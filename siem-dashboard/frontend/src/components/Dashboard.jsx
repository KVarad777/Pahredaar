import React, { useEffect, useState, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { Activity, ShieldAlert, Target, ShieldCheck, Database, Terminal, Play, X, Key, Cpu, Hash, BookOpen, AlertTriangle, Skull, Shield } from 'lucide-react';

const API_BASE = '/api';

export function Dashboard() {
  const [metrics, setMetrics] = useState([]);
  const [coverage, setCoverage] = useState([]);
  const [logs, setLogs] = useState([]);
  const [feedback, setFeedback] = useState([]);
  const [loading, setLoading] = useState(true);

  // Tabs State
  const [activeTab, setActiveTab] = useState('dashboard');
  
  // Log details modal
  const [selectedLog, setSelectedLog] = useState(null);

  // Terminal State
  const [terminalLogs, setTerminalLogs] = useState([]);
  const terminalEndRef = useRef(null);

  const fetchData = async () => {
    try {
      const [metricsRes, coverageRes, logsRes, feedbackRes] = await Promise.all([
        fetch(`${API_BASE}/metrics`),
        fetch(`${API_BASE}/coverage`),
        fetch(`${API_BASE}/logs?limit=50`),
        fetch(`${API_BASE}/feedback`)
      ]);
      
      setMetrics(await metricsRes.json());
      setCoverage(await coverageRes.json());
      setLogs(await logsRes.json());
      setFeedback(await feedbackRes.json());
    } catch (e) {
      console.error("Failed to fetch data:", e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    // Refresh less frequently since we have manual generation now
    const interval = setInterval(fetchData, 15000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (terminalEndRef.current) {
      terminalEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [terminalLogs]);

  const handleStartAttack = async (e) => {
    e.preventDefault();
    setTerminalLogs(['> Initializing Red Team AI Attack sequence...', '> Connecting to Groq LLM...']);
    setActiveTab('logs');

    try {
      await fetch(`${API_BASE}/attack/generate`, {
        method: 'POST',
      });

      // Start SSE connection
      const sse = new EventSource(`${API_BASE}/attack/stream`);
      
      sse.onmessage = (event) => {
        if (event.data === '[DONE]') {
          sse.close();
          setTerminalLogs(prev => [...prev, '> [SYSTEM] Round complete. Disconnecting.', '> [SYSTEM] Refreshing dashboard data...']);
          setTimeout(() => {
            fetchData();
          }, 2000);
          return;
        }
        if (event.data) {
          // Replace escaped newlines if any
          const cleanLine = event.data.replace(/\\n/g, '\n');
          setTerminalLogs(prev => [...prev, cleanLine]);
        }
      };

      sse.onerror = (err) => {
        console.error("SSE Error:", err);
        sse.close();
      };

    } catch (e) {
      console.error("Attack Failed:", e);
      setTerminalLogs(prev => [...prev, `> [ERROR] ${e.message}`]);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen w-full">
        <div className="animate-pulse flex flex-col items-center">
          <Activity className="w-12 h-12 text-primary mb-4 animate-spin-slow" />
          <h2 className="text-xl text-primary font-light">Initializing SIEM Environment...</h2>
        </div>
      </div>
    );
  }

  // Calculate Overview Stats
  const latestMetric = metrics.length > 0 ? metrics[metrics.length - 1] : null;
  const totalScenarios = coverage.length;
  const avgDetection = coverage.length > 0 ? 
    (coverage.reduce((acc, curr) => acc + (curr.detection_rate || 0), 0) / coverage.length) : 0;
  
  // Calculate trends dynamically
  const prevMetric = metrics.length > 1 ? metrics[metrics.length - 2] : null;
  const f1Trend = prevMetric && latestMetric ? ((latestMetric.ensemble_f1 - prevMetric.ensemble_f1) * 100) : 0;
  const fprTrend = prevMetric && latestMetric ? ((latestMetric.blue_fpr - prevMetric.blue_fpr) * 100) : 0;

  const formatTrend = (val) => val > 0 ? `+${val.toFixed(2)}%` : val < 0 ? `${val.toFixed(2)}%` : '0.00%';
  
  // --- SIEM METRICS ---
  const fraudLogs = logs.filter(log => log.is_fraud === 1 || log.is_fraud === 1.0).reverse();
  const vectorMap = {};
  coverage.forEach(s => {
      const cat = s.category || 'unknown';
      vectorMap[cat] = (vectorMap[cat] || 0) + 1;
  });
  const topVectors = Object.entries(vectorMap).sort((a,b) => b[1] - a[1]).slice(0, 3);
  
  return (
    <div className="flex h-screen overflow-hidden font-sans relative">
      {/* Sidebar */}
      <aside className="w-64 glass border-r flex flex-col relative z-20">
        <div className="p-6 border-b border-border flex items-center space-x-3">
          <ShieldAlert className="w-8 h-8 text-primary" />
          <h1 className="text-xl font-semibold tracking-tight text-white/90">PAHREDAAR</h1>
        </div>
        
        <nav className="flex-1 p-4 space-y-2">
          <NavItem icon={<Activity />} label="Dashboard" active={activeTab === 'dashboard'} onClick={() => setActiveTab('dashboard')} />
          <NavItem icon={<Target />} label="Coverage Matrix" active={activeTab === 'coverage'} onClick={() => setActiveTab('coverage')} />
          <NavItem icon={<Database />} label="Datasets" active={activeTab === 'datasets'} onClick={() => setActiveTab('datasets')} />
          <NavItem icon={<Cpu />} label="AI Feedback" active={activeTab === 'feedback'} onClick={() => setActiveTab('feedback')} />
          <NavItem icon={<Terminal />} label="Live Logs" active={activeTab === 'logs'} onClick={() => setActiveTab('logs')} />
          <div className="pt-4 pb-2">
            <div className="text-xs font-semibold text-white/30 uppercase tracking-wider px-4">Documentation</div>
          </div>
          <NavItem icon={<BookOpen />} label="About Project" active={activeTab === 'about'} onClick={() => setActiveTab('about')} />
        </nav>
        
        <div className="p-4 border-t border-border">
          <div className="flex items-center space-x-2 text-sm text-green-400 bg-green-400/10 p-2 rounded-lg border border-green-400/20">
            <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse"></div>
            <span>System Active</span>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-y-auto relative z-10 p-8 space-y-8">
        
        {/* Header */}
        <header className="flex justify-between items-end">
          <div>
            <h2 className="text-3xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white to-white/50">
              Red vs Blue Simulation
            </h2>
            <p className="text-white/50 mt-1">Real-time threat detection and efficacy tracking.</p>
          </div>
          <div className="flex items-center space-x-4">
            <div className="text-sm font-mono text-white/40 glass px-4 py-2 rounded-md">
              Round: {latestMetric ? latestMetric.round : 'N/A'}
            </div>
            <button 
              onClick={handleStartAttack}
              className="flex items-center px-4 py-2 bg-danger/20 hover:bg-danger/30 text-danger border border-danger/30 rounded-lg transition-colors font-medium text-sm shadow-[0_0_15px_rgba(239,68,68,0.15)]"
            >
              <Play className="w-4 h-4 mr-2" /> Trigger AI Attack
            </button>
          </div>
        </header>

        {/* Overview Stats */}
        {activeTab === 'dashboard' && (
          <>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
              <StatCard title="Ensemble F1 Score" value={latestMetric ? (latestMetric.ensemble_f1 * 100).toFixed(1) + '%' : '--'} trend={formatTrend(f1Trend)} />
              <StatCard title="Detection Rate" value={coverage.length > 0 ? `${(avgDetection * 100).toFixed(0)}%` : '--'} trend={coverage.length > 0 ? "Active" : "Awaiting"} isNeutral />
              <StatCard title="False Positive Rate" value={latestMetric ? (latestMetric.blue_fpr * 100).toFixed(2) + '%' : '--'} trend={formatTrend(fprTrend)} reverseTrend />
              <StatCard title="Simulated Scenarios" value={totalScenarios > 0 ? totalScenarios : '--'} trend={totalScenarios > 0 ? "Active" : "Awaiting"} isNeutral />
            </div>

            {/* SIEM Threat Intel Row */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              
              {/* Top Vectors */}
              <div className="glass rounded-xl p-6 relative group overflow-hidden siem-border-danger">
                <div className="absolute inset-0 bg-gradient-to-br from-danger/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                <h3 className="text-lg font-medium mb-6 flex items-center text-danger"><Skull className="w-5 h-5 mr-2" /> Top Threat Vectors</h3>
                <div className="space-y-4">
                  {topVectors.length > 0 ? topVectors.map(([category, count], idx) => (
                    <div key={category} className="flex items-center justify-between p-3 bg-white/5 rounded-lg border border-white/5">
                      <div className="flex items-center space-x-3">
                        <span className="text-sm font-mono text-white/40">0{idx + 1}</span>
                        <span className="font-medium capitalize text-white/80">{category}</span>
                      </div>
                      <span className="text-danger font-mono text-sm bg-danger/10 px-2 py-1 rounded">{count} Scenarios</span>
                    </div>
                  )) : (
                    <div className="text-center text-white/40 py-8 font-mono text-sm">NO THREAT VECTORS DETECTED</div>
                  )}
                </div>
              </div>

              {/* Live Alerts Ticker */}
              <div className="glass rounded-xl p-6 relative group overflow-hidden lg:col-span-2 flex flex-col h-72 siem-border">
                <div className="absolute inset-0 bg-gradient-to-br from-warning/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                <h3 className="text-lg font-medium mb-4 flex items-center text-warning"><AlertTriangle className="w-5 h-5 mr-2" /> Live Security Alerts</h3>
                <div className="flex-1 overflow-auto pr-2 custom-scrollbar">
                  {fraudLogs.length > 0 ? (
                    <table className="w-full text-left border-collapse font-mono text-sm">
                      <thead className="sticky top-0 bg-[#06090e] z-10 text-white/40 text-xs">
                        <tr>
                          <th className="py-2 border-b border-border font-medium">TIMESTAMP / ID</th>
                          <th className="py-2 border-b border-border font-medium text-right">AMOUNT (₹)</th>
                          <th className="py-2 border-b border-border font-medium px-4">SCENARIO FLAG</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-border">
                        {fraudLogs.slice(0, 15).map((log, i) => (
                          <tr key={i} className="hover:bg-white/5 transition-colors cursor-pointer group" onClick={() => setSelectedLog(log)}>
                            <td className="py-3 text-danger group-hover:text-danger/80">
                              <div>{log.timestamp || 'N/A'}</div>
                              <div className="text-xs text-white/30">{log.transaction_id ? log.transaction_id.substring(0, 8) : 'N/A'}</div>
                            </td>
                            <td className="py-3 text-right text-white/70">{(log.amount || 0).toLocaleString()}</td>
                            <td className="py-3 px-4">
                              <span className="bg-danger/10 text-danger border border-danger/20 px-2 py-1 rounded text-xs truncate max-w-[300px] inline-block" title={log.scenario_name || 'Generic Fraud'}>
                                {log.scenario_name || 'Generic Fraud'}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  ) : (
                    <div className="flex flex-col items-center justify-center h-full text-white/40 space-y-3 font-mono text-sm">
                      <Shield className="w-10 h-10 text-primary/30" />
                      <div>SYSTEM SECURE - NO ALERTS DETECTED</div>
                    </div>
                  )}
                </div>
              </div>

            </div>

            {/* Charts Row */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="glass rounded-xl p-6 relative group overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-br from-primary/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
            <h3 className="text-lg font-medium mb-6 flex items-center"><Activity className="w-5 h-5 mr-2 text-primary" /> Detection Efficacy over Rounds</h3>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={metrics}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" vertical={false} />
                  <XAxis dataKey="round" stroke="rgba(255,255,255,0.4)" fontSize={12} tickLine={false} axisLine={false} />
                  <YAxis stroke="rgba(255,255,255,0.4)" fontSize={12} tickLine={false} axisLine={false} domain={[0, 1]} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#1E293B', borderColor: 'rgba(255,255,255,0.1)', borderRadius: '8px' }}
                    itemStyle={{ color: '#E2E8F0' }}
                  />
                  <Legend iconType="circle" wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
                  <Line type="monotone" dataKey="ensemble_recall" name="Recall" stroke="#3B82F6" strokeWidth={3} dot={{ r: 4 }} activeDot={{ r: 6 }} />
                  <Line type="monotone" dataKey="ensemble_precision" name="Precision" stroke="#10B981" strokeWidth={3} dot={{ r: 4 }} />
                  <Line type="monotone" dataKey="ensemble_f1" name="F1 Score" stroke="#F59E0B" strokeWidth={3} dot={{ r: 4 }} />
                  <Line type="monotone" dataKey="blue_fpr" name="FPR" stroke="#EF4444" strokeWidth={2} strokeDasharray="5 5" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="glass rounded-xl p-6 relative group overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-bl from-secondary/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
            <h3 className="text-lg font-medium mb-6 flex items-center"><Target className="w-5 h-5 mr-2 text-secondary" /> Red vs Blue Reward Signal</h3>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={metrics}>
                  <defs>
                    <linearGradient id="colorBlue" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#3B82F6" stopOpacity={0}/>
                    </linearGradient>
                    <linearGradient id="colorRed" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#EF4444" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#EF4444" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" vertical={false} />
                  <XAxis dataKey="round" stroke="rgba(255,255,255,0.4)" fontSize={12} tickLine={false} axisLine={false} />
                  <YAxis stroke="rgba(255,255,255,0.4)" fontSize={12} tickLine={false} axisLine={false} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#1E293B', borderColor: 'rgba(255,255,255,0.1)', borderRadius: '8px' }}
                  />
                  <Legend iconType="circle" />
                  <Area type="monotone" dataKey="blue_reward" name="Blue Reward" stroke="#3B82F6" fillOpacity={1} fill="url(#colorBlue)" />
                  <Area type="monotone" dataKey="mean_red_reward" name="Mean Red Reward" stroke="#EF4444" fillOpacity={1} fill="url(#colorRed)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
          </div>
          </>
        )}

        {/* Matrices and Logs */}
        {(activeTab === 'dashboard' || activeTab === 'coverage' || activeTab === 'datasets') && (
          <div className={`grid grid-cols-1 ${activeTab === 'dashboard' ? 'xl:grid-cols-2' : ''} gap-6`}>
            {/* Coverage Matrix Table */}
            {(activeTab === 'dashboard' || activeTab === 'coverage') && (
              <div className={`glass rounded-xl p-6 flex flex-col ${activeTab === 'coverage' ? 'h-[700px]' : 'h-[500px]'}`}>
            <h3 className="text-lg font-medium mb-4 flex items-center"><Target className="w-5 h-5 mr-2 text-primary" /> Scenario Coverage Matrix</h3>
            <div className="flex-1 overflow-auto pr-2 custom-scrollbar">
              <table className="w-full text-left border-collapse">
                <thead className="sticky top-0 bg-[#161925] z-10 shadow-md">
                  <tr>
                    <th className="py-3 px-4 text-xs font-semibold text-white/50 uppercase tracking-wider border-b border-border">Scenario</th>
                    <th className="py-3 px-4 text-xs font-semibold text-white/50 uppercase tracking-wider border-b border-border">Category</th>
                    <th className="py-3 px-4 text-xs font-semibold text-white/50 uppercase tracking-wider border-b border-border">Detection</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {coverage.map((c, i) => (
                    <tr key={i} className="hover:bg-white/[0.02] transition-colors group">
                      <td className="py-3 px-4 font-medium text-sm group-hover:text-primary transition-colors">{c.scenario_name}</td>
                      <td className="py-3 px-4 text-sm text-white/70">
                        <span className="px-2 py-1 bg-white/5 rounded-md text-xs border border-border">
                          {c.manipulation_type}
                        </span>
                      </td>
                      <td className="py-3 px-4">
                        <div className="flex items-center space-x-2">
                          <div className="w-full bg-black/40 rounded-full h-2 max-w-[100px] overflow-hidden">
                            <div 
                              className={`h-2 rounded-full ${c.detection_rate > 0.5 ? 'bg-secondary' : 'bg-danger'}`} 
                              style={{ width: `${(c.detection_rate || 0) * 100}%` }}
                            ></div>
                          </div>
                          <span className="text-xs text-white/60 w-8">{((c.detection_rate || 0) * 100).toFixed(0)}%</span>
                        </div>
                      </td>
                    </tr>
                  ))}
                  {coverage.length === 0 && (
                     <tr>
                        <td colSpan="3" className="py-8 text-center text-white/40">No scenarios found</td>
                     </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
            )}

          {/* Live Logs Table */}
          {(activeTab === 'dashboard' || activeTab === 'datasets') && (
            <div className={`glass rounded-xl p-6 flex flex-col ${activeTab === 'datasets' ? 'h-[700px]' : 'h-[500px]'}`}>
            <h3 className="text-lg font-medium mb-4 flex items-center justify-between">
              <div className="flex items-center"><Terminal className="w-5 h-5 mr-2 text-secondary" /> Synthetic Event Logs</div>
              <div className="text-xs bg-secondary/20 text-secondary px-2 py-1 rounded-md border border-secondary/20 animate-pulse">Live</div>
            </h3>
            <div className="flex-1 overflow-auto pr-2 custom-scrollbar">
              <table className="w-full text-left border-collapse">
                <thead className="sticky top-0 bg-[#161925] z-10 shadow-md">
                  <tr>
                    <th className="py-3 px-4 text-xs font-semibold text-white/50 uppercase tracking-wider border-b border-border">Time / ID</th>
                    <th className="py-3 px-4 text-xs font-semibold text-white/50 uppercase tracking-wider border-b border-border">Amount</th>
                    <th className="py-3 px-4 text-xs font-semibold text-white/50 uppercase tracking-wider border-b border-border">Type</th>
                    <th className="py-3 px-4 text-xs font-semibold text-white/50 uppercase tracking-wider border-b border-border">Blue Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {logs.map((log, i) => (
                    <tr 
                      key={i} 
                      className={`hover:bg-white/[0.04] transition-colors cursor-pointer ${log.is_fraud === 1 ? 'bg-danger/5' : ''}`}
                      onClick={() => setSelectedLog(log)}
                    >
                      <td className="py-3 px-4 text-sm font-mono">
                        <div className="text-white/80">{log.step || log.time || `T-${i}`}</div>
                        <div className="text-xs text-white/40 truncate w-24" title={log.nameOrig}>{log.nameOrig || 'Unknown'}</div>
                      </td>
                      <td className="py-3 px-4 text-sm font-medium">
                        ${Number(log.amount).toFixed(2)}
                      </td>
                      <td className="py-3 px-4 text-sm">
                        {log.is_fraud === 1 ? (
                           <span className="px-2 py-1 bg-danger/20 text-danger rounded-md text-xs border border-danger/20 font-medium">Attack</span>
                        ) : (
                           <span className="px-2 py-1 bg-white/5 text-white/60 rounded-md text-xs border border-border">Legit</span>
                        )}
                      </td>
                      <td className="py-3 px-4 text-sm">
                         {log.is_fraud === 1 ? (
                            <div className="flex items-center text-secondary">
                              <ShieldCheck className="w-4 h-4 mr-1" /> <span className="text-xs">Blocked</span>
                            </div>
                         ) : (
                            <div className="flex items-center text-white/40">
                              <span className="text-xs">Allowed</span>
                            </div>
                         )}
                      </td>
                    </tr>
                  ))}
                  {logs.length === 0 && (
                     <tr>
                        <td colSpan="4" className="py-8 text-center text-white/40">No logs found</td>
                     </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
            )}
          </div>
        )}

        {/* Feedback Engine View */}
        {activeTab === 'feedback' && (
          <div className="space-y-6 animate-fade-in">
            <div className="flex items-center space-x-3 mb-6">
              <Cpu className="w-8 h-8 text-secondary" />
              <div>
                <h3 className="text-xl font-medium tracking-tight">AI Red Team Feedback Engine</h3>
                <p className="text-sm text-white/50">Miss explanations and plain-language tactics fed back into the Identity Engine.</p>
              </div>
            </div>
            
            {feedback.length > 0 ? (
              <div className="grid gap-4">
                {feedback.map((exp, i) => (
                  <div key={i} className="glass rounded-xl p-6 border-l-4 border-secondary/50 hover:border-secondary transition-colors relative overflow-hidden group">
                    <div className="absolute inset-0 bg-gradient-to-r from-secondary/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                    <p className="text-white/80 leading-relaxed font-mono text-sm relative z-10">{exp}</p>
                  </div>
                ))}
              </div>
            ) : (
              <div className="glass rounded-xl p-12 text-center text-white/40 flex flex-col items-center justify-center">
                <ShieldCheck className="w-12 h-12 mb-4 opacity-20" />
                <p>No feedback available yet.</p>
                <p className="text-sm mt-2">Run an attack round to generate miss explanations.</p>
              </div>
            )}
          </div>
        )}

        {/* Embedded Terminal View */}
        {activeTab === 'logs' && (
          <div className="h-[700px] bg-[#0A0A0A] border border-[#333] rounded-xl shadow-2xl flex flex-col overflow-hidden animate-fade-in">
            <div className="bg-[#1A1A1A] p-3 flex justify-between items-center border-b border-[#333]">
              <div className="flex items-center space-x-2">
                <Terminal className="w-4 h-4 text-green-400" />
                <span className="text-xs font-mono text-white/60">sys.stdout - FraudRedTeamLoop</span>
              </div>
              <div className="flex space-x-2">
                 <div className="w-3 h-3 rounded-full bg-[#EF4444]/50 border border-[#EF4444]/20"></div>
                 <div className="w-3 h-3 rounded-full bg-[#F59E0B]/50 border border-[#F59E0B]/20"></div>
                 <div className="w-3 h-3 rounded-full bg-[#10B981]/50 border border-[#10B981]/20"></div>
              </div>
            </div>
            
            <div className="flex-1 p-6 font-mono text-sm overflow-y-auto custom-scrollbar bg-[#0A0A0A]">
              {terminalLogs.length > 0 ? terminalLogs.map((log, idx) => (
                <div key={idx} className="mb-1 text-gray-300">
                  <span className="text-green-500/50 mr-2">$</span>
                  <span style={{ whiteSpace: 'pre-wrap' }}>{log}</span>
                </div>
              )) : (
                <div className="text-white/20 italic">Waiting for terminal output... click "Trigger AI Attack" to begin.</div>
              )}
              <div ref={terminalEndRef} />
            </div>
          </div>
        )}

        {/* About Project Tab */}
        {activeTab === 'about' && (
          <div className="max-w-4xl space-y-8 animate-fade-in">
            <div className="flex items-center space-x-3 mb-8">
              <BookOpen className="w-8 h-8 text-primary" />
              <div>
                <h3 className="text-2xl font-bold tracking-tight">Project Documentation</h3>
                <p className="text-white/50">Evaluation criteria and architectural approach.</p>
              </div>
            </div>

            <div className="glass rounded-xl p-8 space-y-6">
              <h2 className="text-xl font-semibold text-white border-b border-border pb-4">1. Diversity of attacks identified</h2>
              
              <div className="space-y-4 text-white/80 leading-relaxed">
                <p>
                  <strong className="text-primary">What it's really asking:</strong> not "did you build one fraud simulator" but "how many genuinely different <em>mechanisms</em> of fraud can your system produce and reason about."
                </p>
                
                <h4 className="text-lg font-medium text-white pt-4">How to score well:</h4>
                <ul className="list-disc pl-5 space-y-2 text-sm">
                  <li>Don't just vary parameters of one attack (amount, location) — cover distinct <em>categories</em> of mechanism:</li>
                  <ul className="list-circle pl-5 space-y-1 text-white/60">
                    <li>Identity-based (synthetic identity, account takeover via social engineering)</li>
                    <li>Behavior-based (low-and-slow, velocity abuse)</li>
                    <li>Network-based (mule networks, device/IP sharing rings, collusive merchants)</li>
                    <li>Channel-based (card-not-present abuse, promo/refund abuse, chargeback fraud)</li>
                    <li>AI-specific (deepfake-assisted KYC bypass, AI-generated phishing leading to compromised credentials)</li>
                  </ul>
                  <li>Keep a visible "coverage matrix" in your Identify engine's output — e.g. a table of scenario name × category × novelty tag. Judges can see breadth at a glance instead of having to infer it.</li>
                  <li>Aim for <strong className="text-secondary">quality over sheer count</strong>: 8–10 clearly distinct mechanisms beats 30 parameter variations of the same 2 ideas.</li>
                </ul>
              </div>
            </div>
          </div>
        )}

      </main>

      {/* Log Details Modal */}
      {selectedLog && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4 animate-fade-in">
          <div className="w-full max-w-2xl bg-[#0F1219] border border-[#333] rounded-2xl shadow-2xl overflow-hidden flex flex-col">
            <div className="bg-[#161925] p-5 flex justify-between items-center border-b border-[#333]">
              <h3 className="text-lg font-medium flex items-center">
                <Database className="w-5 h-5 mr-2 text-primary" />
                Transaction Details
              </h3>
              <button onClick={() => setSelectedLog(null)} className="text-white/40 hover:text-white transition-colors">
                <X className="w-5 h-5" />
              </button>
            </div>
            
            <div className="p-6 overflow-y-auto max-h-[70vh] custom-scrollbar">
              {selectedLog.is_fraud === 1 && (
                <div className="mb-6 p-4 bg-danger/10 border border-danger/20 rounded-xl flex items-start space-x-3">
                  <ShieldAlert className="w-5 h-5 text-danger shrink-0 mt-0.5" />
                  <div>
                    <h4 className="text-sm font-semibold text-danger">AI Attack Scenario</h4>
                    <p className="text-xs text-danger/80 mt-1">This transaction was synthetically generated by the Red Team LLM engine.</p>
                  </div>
                </div>
              )}
              
              <div className="grid grid-cols-2 gap-4">
                {Object.entries(selectedLog).map(([key, value]) => (
                  <div key={key} className="glass p-3 rounded-lg border border-white/5">
                    <div className="text-xs text-white/40 uppercase tracking-wider mb-1">{key}</div>
                    <div className="text-sm font-mono truncate" title={String(value)}>
                      {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}

function NavItem({ icon, label, active, onClick }) {
  return (
    <div 
      onClick={onClick}
      className={`flex items-center space-x-3 px-4 py-3 rounded-xl cursor-pointer transition-all duration-200
      ${active ? 'bg-primary/20 text-primary border border-primary/20 shadow-[0_0_15px_rgba(59,130,246,0.15)]' : 'text-white/60 hover:text-white hover:bg-white/5 border border-transparent'}`}>
      <div className="opacity-80">{icon}</div>
      <span className="font-medium text-sm">{label}</span>
    </div>
  );
}

function StatCard({ title, value, trend, isNeutral, reverseTrend }) {
  const isPositive = trend?.startsWith('+');
  const trendColor = isNeutral ? 'text-white/40' : 
                     reverseTrend ? (isPositive ? 'text-danger' : 'text-secondary') : 
                     (isPositive ? 'text-secondary' : 'text-danger');
                     
  return (
    <div className="glass rounded-xl p-5 relative overflow-hidden group">
      <div className="absolute inset-0 bg-gradient-to-br from-white/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
      <h4 className="text-sm font-medium text-white/50 mb-1">{title}</h4>
      <div className="flex items-end justify-between">
        <div className="text-3xl font-bold tracking-tight">{value}</div>
        <div className={`text-xs font-semibold ${trendColor} bg-black/20 px-2 py-1 rounded-md`}>
          {trend}
        </div>
      </div>
    </div>
  );
}
