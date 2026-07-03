// App.jsx - Component Configuration
import React, { useState } from 'react';

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [report, setReport] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setReport(null);
      setError(null);
    }
  };

  const handleInferenceSubmit = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('http://localhost:8000/api/v1/diagnose', {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();
      if (response.ok && result.success) {
        setReport(result.data);
      } else {
        setError(result.detail || 'Analysis validation checkpoint failed.');
      }
    } catch (err) {
      setError('Could not connect to the remote machine learning endpoint.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 font-sans antialiased">
      {/* Navbar Container */}
      <nav className="border-b border-slate-800 bg-slate-900/50 backdrop-blur sticky top-0 z-50 px-6 py-4">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <div className="flex items-center space-x-3">
            <span className="text-xl font-bold tracking-tight text-emerald-400">AgTech Vision OS</span>
          </div>
          <div className="text-xs bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-3 py-1 rounded-full uppercase tracking-wider font-semibold">
            API System Active
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-6 py-10 grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Side: Upload & Stage Panel */}
        <div className="lg:col-span-5 space-y-6">
          <div className="bg-slate-800/40 border border-slate-800 rounded-2xl p-6 backdrop-blur">
            <h2 className="text-lg font-semibold mb-4 text-slate-200">Ingest Specimen Asset</h2>
            
            <label className="flex flex-col items-center justify-center w-full h-64 border-2 border-dashed border-slate-700 hover:border-emerald-500/50 rounded-xl cursor-pointer transition bg-slate-900/40 group overflow-hidden relative">
              {previewUrl ? (
                <img src={previewUrl} alt="Preview" className="w-full h-full object-cover" />
              ) : (
                <div className="flex flex-col items-center justify-center pt-5 pb-6 px-4 text-center">
                  <p className="mb-2 text-sm text-slate-400">
                    <span className="font-semibold text-emerald-400 group-hover:underline">Click to upload file</span> or drag and drop
                  </p>
                  <p className="text-xs text-slate-500">Supports high-res PNG, JPG or JPEG leaf viewports</p>
                </div>
              )}
              <input type="file" className="hidden" accept="image/*" onChange={handleFileChange} />
            </label>

            {selectedFile && (
              <button
                onClick={handleInferenceSubmit}
                disabled={loading}
                className="mt-4 w-full bg-emerald-500 hover:bg-emerald-600 disabled:bg-slate-700 disabled:cursor-not-allowed text-slate-950 font-semibold py-3 px-4 rounded-xl transition flex justify-center items-center space-x-2"
              >
                {loading ? (
                  <span>Executing Pipeline Matrix...</span>
                ) : (
                  <span>Run Diagnostics</span>
                )}
              </button>
            )}
          </div>

          {error && (
            <div className="bg-rose-500/10 border border-rose-500/20 text-rose-400 rounded-xl p-4 text-sm font-medium">
              {error}
            </div>
          )}
        </div>

        {/* Right Side: Evaluation Matrix Reports */}
        <div className="lg:col-span-7">
          {report ? (
            <div className="bg-slate-800/40 border border-slate-800 rounded-2xl p-8 backdrop-blur space-y-6">
              <div className="flex justify-between items-start border-b border-slate-800 pb-5">
                <div>
                  <h3 className="text-xs uppercase font-semibold text-slate-400 tracking-wider">Diagnostic Evaluation Result</h3>
                  <h2 className="text-2xl font-bold text-slate-100 mt-1">{report.crop} Model Profile</h2>
                </div>
                <span className={`px-4 py-1.5 rounded-full text-xs font-bold uppercase tracking-wider border ${
                  report.condition === 'HEALTHY' 
                    ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' 
                    : 'bg-amber-500/10 text-amber-400 border-amber-500/20'
                }`}>
                  {report.condition}
                </span>
              </div>

              <div className="grid grid-cols-2 gap-6">
                <div className="bg-slate-900/60 rounded-xl p-4 border border-slate-800">
                  <span className="text-xs font-medium text-slate-500 uppercase tracking-wider">Identified Pathology</span>
                  <p className="text-lg font-semibold text-slate-200 mt-1 capitalize">{report.diagnosed_class}</p>
                </div>
                <div className="bg-slate-900/60 rounded-xl p-4 border border-slate-800">
                  <span className="text-xs font-medium text-slate-500 uppercase tracking-wider">Surface Area Severity</span>
                  <div className="flex items-baseline space-x-2 mt-1">
                    <p className="text-2xl font-bold text-emerald-400">{report.severity_percentage}%</p>
                  </div>
                </div>
              </div>

              {/* Severity Bar Widget */}
              <div className="space-y-2">
                <div className="flex justify-between text-xs font-medium text-slate-400">
                  <span>Necrosis Progression Matrix</span>
                  <span>{report.severity_percentage}% Index</span>
                </div>
                <div className="w-full bg-slate-900 rounded-full h-2.5 overflow-hidden border border-slate-800">
                  <div 
                    className="bg-emerald-400 h-2.5 rounded-full transition-all duration-500" 
                    style={{ width: `${report.severity_percentage}%` }}
                  />
                </div>
              </div>

              <div className="border-t border-slate-800 pt-5 space-y-3">
                <h4 className="text-xs font-semibold uppercase text-slate-400 tracking-wider">Actionable Agrarian Protocols</h4>
                <div className="bg-emerald-500/5 border border-emerald-500/10 rounded-xl p-5 text-sm leading-relaxed text-slate-300">
                  {report.treatment_protocol}
                </div>
              </div>
            </div>
          ) : (
            <div className="h-full min-h-[400px] border border-dashed border-slate-800 rounded-2xl flex flex-col items-center justify-center text-center p-6 bg-slate-900/10">
              <p className="text-slate-400 font-medium">System Awaiting Execution Pipeline Run</p>
              <p className="text-xs text-slate-600 max-w-xs mt-1">Ingest a standard leaf viewport matrix asset on the left panel to build full diagnostic outputs.</p>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}