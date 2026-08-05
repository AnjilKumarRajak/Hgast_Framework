"use client"

import { useState, useRef } from "react"
import { WebGLShader } from "@/components/ui/web-gl-shader";
import { LiquidButton } from '@/components/ui/liquid-glass-button' 

export default function DemoOne() {
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState<any>(null)
  const [isRecording, setIsRecording] = useState(false)
  
  const mediaRecorder = useRef<MediaRecorder | null>(null)
  const audioChunks = useRef<Blob[]>([])

  const processAudioFile = async (fileOrBlob: Blob | File, filename: string) => {
    setLoading(true)
    setResults(null)
    
    const formData = new FormData()
    formData.append("audio", fileOrBlob, filename)
    
    try {
        const response = await fetch("/api/translate", {
            method: "POST",
            body: formData,
        })
        const data = await response.json()
        setResults(data)
    } catch (err) {
        console.error("Error connecting to backend:", err)
        alert("Failed to connect to the FastAPI backend.")
    } finally {
        setLoading(false)
    }
  }
  
  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return
    processAudioFile(file, file.name)
  }

  const toggleRecording = async () => {
    if (isRecording) {
      // Stop recording
      mediaRecorder.current?.stop()
      setIsRecording(false)
    } else {
      // Start recording
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
        const recorder = new MediaRecorder(stream)
        mediaRecorder.current = recorder
        audioChunks.current = []

        recorder.ondataavailable = (event) => {
          if (event.data.size > 0) {
            audioChunks.current.push(event.data)
          }
        }

        recorder.onstop = () => {
          const audioBlob = new Blob(audioChunks.current, { type: 'audio/webm' })
          processAudioFile(audioBlob, "recording.webm")
          // Stop all tracks to release microphone
          stream.getTracks().forEach(track => track.stop())
        }

        recorder.start()
        setIsRecording(true)
      } catch (err) {
        console.error("Microphone access denied:", err)
        alert("Please allow microphone access to record audio.")
      }
    }
  }

  return (
    <div className="relative flex min-h-screen w-full flex-col items-center justify-center overflow-hidden bg-black p-4">
      <WebGLShader/> 
      <div className="relative border border-[#27272a] p-2 w-full mx-auto max-w-4xl z-10 bg-black/40 backdrop-blur-sm rounded-xl">
        <main className="relative border border-[#27272a] py-12 px-6 overflow-hidden rounded-lg">
          <h1 className="mb-4 text-white text-center text-6xl font-extrabold tracking-tighter md:text-[clamp(2rem,7vw,6rem)]">HGAST</h1>
          <p className="text-white/90 px-8 text-center text-lg md:text-xl font-semibold max-w-2xl mx-auto mb-2">Hallucination-free Gender-Aware Speech Translation.</p>
          <p className="text-white/60 px-8 text-center text-sm md:text-base italic max-w-2xl mx-auto">"Correcting morphological bias in state-of-the-art foundation models."</p>
          <div className="my-8 flex items-center justify-center gap-2">
              <span className="relative flex h-3 w-3 items-center justify-center">
                  <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-green-500 opacity-75"></span>
                  <span className="relative inline-flex h-2 w-2 rounded-full bg-green-500"></span>
              </span>
              <p className="text-xs text-green-500 font-semibold tracking-wider uppercase">Models Loaded</p>
          </div>
                
          <div className="flex justify-center gap-4 mt-8 relative"> 
              <div onClick={toggleRecording} className={`cursor-pointer ${loading ? 'opacity-50 pointer-events-none' : ''}`}>
                <LiquidButton className={`text-white border-white/20 rounded-full hover:bg-white/10 ${isRecording ? 'bg-red-500/20 hover:bg-red-500/30 border-red-500/50' : 'bg-white/5'}`} size={'xl'}>
                  {isRecording ? "Stop Recording" : "Record Audio"}
                  {isRecording && <span className="ml-2 h-2 w-2 rounded-full bg-red-500 animate-pulse"></span>}
                </LiquidButton> 
              </div>
              <div className={`relative ${loading ? 'opacity-50 pointer-events-none' : ''}`}>
                <input type="file" accept="audio/*" onChange={handleFileUpload} className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-50" disabled={loading} />
                <LiquidButton className="text-white border-white/20 rounded-full bg-white/5 hover:bg-white/10" size={'xl'}>Upload File</LiquidButton> 
              </div>
          </div>
          
          {/* Animated Processing Bar */}
          {loading && (
              <div className="mt-12 max-w-xl mx-auto space-y-3">
                  <div className="flex justify-between items-center px-1">
                      <p className="text-sm font-semibold tracking-widest uppercase text-white/70 animate-pulse">Processing Audio</p>
                      <p className="text-xs text-white/40 animate-pulse">Running dual-grammar controller...</p>
                  </div>
                  <div className="h-1.5 w-full bg-white/10 rounded-full overflow-hidden">
                      <div className="h-full bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 w-[40%] rounded-full animate-[progress_1.5s_ease-in-out_infinite_alternate]"></div>
                  </div>
                  <style>{`
                    @keyframes progress {
                        0% { width: 0%; transform: translateX(-10%); }
                        100% { width: 40%; transform: translateX(160%); }
                    }
                  `}</style>
              </div>
          )}
          
          {results && results.error && !loading && (
              <div className="mt-12 bg-red-500/10 border border-red-500/30 rounded-lg p-6 max-w-4xl w-full mx-auto space-y-4">
                  <h3 className="text-red-400 font-bold flex items-center gap-2">❌ Backend API Error</h3>
                  <p className="text-red-300 font-mono text-sm">{results.error}</p>
              </div>
          )}
          
          {results && !results.error && !loading && (
              <div className="mt-12 bg-white/5 border border-white/10 rounded-lg p-6 max-w-4xl w-full mx-auto space-y-6">
                  
                  {/* Final Results */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-2 p-4 bg-black/40 rounded-lg border border-red-500/20">
                        <h3 className="text-white/60 text-xs font-bold uppercase tracking-wider mb-1 flex items-center gap-2"><span className="text-red-400">❌</span> Baseline Translation</h3>
                        <p className="text-white/90 text-lg">{results.baseline_translation}</p>
                    </div>
                    <div className="space-y-2 p-4 bg-black/40 rounded-lg border border-green-500/20 shadow-[0_0_15px_rgba(34,197,94,0.1)]">
                        <h3 className="text-green-400 text-xs font-bold uppercase tracking-wider mb-1 flex items-center gap-2"><span className="text-green-400">✅</span> HGAST Corrected Translation</h3>
                        <p className="text-green-300 text-xl font-bold">{results.corrected_translation}</p>
                        
                        {/* Audio Player for Synthesized Speech */}
                        {results.audio_base64 && (
                            <div className="mt-4 pt-4 border-t border-green-500/20">
                                <p className="text-xs text-green-400/80 mb-2 uppercase tracking-wider font-semibold">Synthesized Hindi Speech</p>
                                <audio controls src={`data:audio/wav;base64,${results.audio_base64}`} className="w-full h-10 rounded-full bg-black" autoPlay />
                            </div>
                        )}
                    </div>
                  </div>

                  {/* Intermediate Analysis */}
                  {results.intermediate && (
                  <div className="pt-6 border-t border-white/10">
                      <h3 className="text-white/60 text-xs font-bold uppercase tracking-wider mb-4 flex items-center gap-2">🧠 HGAST Intermediate Analysis (DualGrammarController)</h3>
                      
                      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                        <div className="bg-white/5 p-3 rounded-md border border-white/5">
                            <p className="text-[10px] text-white/40 uppercase tracking-widest mb-1">English ASR</p>
                            <p className="text-white/90 text-sm">{results.intermediate.english_asr || "N/A"}</p>
                        </div>
                        <div className="bg-white/5 p-3 rounded-md border border-white/5">
                            <p className="text-[10px] text-white/40 uppercase tracking-widest mb-1">Speaker Gender</p>
                            <p className="text-blue-300 text-sm font-semibold">{results.intermediate.speaker_gender || "N/A"}</p>
                        </div>
                        <div className="bg-white/5 p-3 rounded-md border border-white/5">
                            <p className="text-[10px] text-white/40 uppercase tracking-widest mb-1">Grammatical Person</p>
                            <p className="text-purple-300 text-sm font-semibold">{results.intermediate.grammatical_person || "N/A"}</p>
                        </div>
                        <div className="bg-white/5 p-3 rounded-md border border-white/5">
                            <p className="text-[10px] text-white/40 uppercase tracking-widest mb-1">Subject Gender</p>
                            <p className="text-yellow-300 text-sm font-semibold">{results.intermediate.subject_gender || "N/A"}</p>
                        </div>
                        <div className="bg-white/5 p-3 rounded-md border border-white/5">
                            <p className="text-[10px] text-white/40 uppercase tracking-widest mb-1">Target Grammatical Gender</p>
                            <p className="text-green-300 text-sm font-semibold uppercase">{results.intermediate.target_gender || "N/A"}</p>
                        </div>
                        <div className="bg-white/5 p-3 rounded-md border border-white/5">
                            <p className="text-[10px] text-white/40 uppercase tracking-widest mb-1">Dominant Controller</p>
                            <p className="text-white/90 text-sm font-mono">{results.intermediate.dominant_controller || "N/A"}</p>
                        </div>
                      </div>
                  </div>
                  )}
              </div>
          )}
        </main>
      </div>
    </div>
  )
}
