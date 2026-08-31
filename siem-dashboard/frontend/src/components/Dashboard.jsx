import React, { useEffect, useState, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { Activity, ShieldAlert, Target, ShieldCheck, Database, Terminal, Play, X, Key, Cpu, Hash } from 'lucide-react';

const API_BASE = 'http://localhost:8000/api';

export function Dashboard() {
  const [metrics, setMetrics] = useState([]);
  const [coverage, setCoverage] = useState([]);
  const [logs, setLogs] = useState([]);
  const [feedback, setFeedback] = useState([]);
  const [loading, setLoading] = useState(true);

  // Tabs State
  const [activeTab, setActiveTab] = useState('dashboard');

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
              <StatCard title="Ensemble F1 Score" value={latestMetric ? (latestMetric.ensemble_f1 * 100).toFixed(1) + '%' : '--'} trend="+2.4%" />
              <StatCard title="Detection Rate" value={`${(avgDetection * 100).toFixed(0)}%`} trend="+5.1%" />
              <StatCard title="False Positive Rate" value={latestMetric ? (latestMetric.blue_fpr * 100).toFixed(2) + '%' : '--'} trend="-0.5%" reverseTrend />
              <StatCard title="Simulated Scenarios" value={totalScenarios} trend="Active" isNeutral />
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
                    <tr key={i} className={`hover:bg-white/[0.02] transition-colors ${log.is_fraud === 1 ? 'bg-danger/5' : ''}`}>
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
      </main>

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
