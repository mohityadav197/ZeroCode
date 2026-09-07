import { useEffect, useRef, useState } from 'react'
import { motion } from 'framer-motion'
import { MessageCircle, Send, User } from 'lucide-react'
import { sendChat } from '../services/api'

function formatTime(date) {
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

function ZAvatar() {
  return (
    <div className="w-7 h-7 rounded-full bg-gradient-to-br from-indigo-500 to-purple-500 flex items-center justify-center shrink-0 text-white text-xs font-extrabold">
      Z
    </div>
  )
}

function ChatPanel({ analysisResult, mlResult, sessionId, initialMessages }) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const endRef = useRef(null)

  useEffect(() => {
    setMessages(
      initialMessages?.length
        ? initialMessages.map((m) => ({
            role: m.role,
            content: m.message,
            time: m.created_at ? new Date(m.created_at) : new Date(),
          }))
        : []
    )
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId])

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const canChat = Boolean(analysisResult)

  const handleSend = async () => {
    const text = input.trim()
    if (!text || loading) return

    setMessages((prev) => [...prev, { role: 'user', content: text, time: new Date() }])
    setInput('')
    setLoading(true)

    try {
      const data = await sendChat(text, sessionId)
      setMessages((prev) => [...prev, { role: 'assistant', content: data.response, time: new Date() }])
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: 'Sorry, something went wrong reaching ZeroCode AI.', time: new Date() },
      ])
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="flex flex-col h-full glass-card rounded-none sm:rounded-2xl">
      <div className="px-4 py-3 border-b border-white/10 shrink-0 flex items-center gap-2">
        <ZAvatar />
        <h2 className="font-semibold text-slate-100">ZeroCode Assistant</h2>
        <div className="relative w-2.5 h-2.5 ml-auto">
          <span className="absolute inset-0 rounded-full bg-emerald-400 animate-ping" />
          <span className="absolute inset-0 rounded-full bg-emerald-400" />
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-3">
        {!canChat && messages.length === 0 && (
          <div className="flex flex-col items-center justify-center text-center mt-12 gap-3">
            <div className="w-14 h-14 rounded-full bg-white/5 border border-white/10 flex items-center justify-center">
              <MessageCircle className="w-6 h-6 text-slate-500" />
            </div>
            <p className="text-sm text-slate-500 max-w-[220px]">
              Upload a CSV and run analysis to start chatting with your data
            </p>
          </div>
        )}

        {messages.map((msg, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.25 }}
            className={`flex items-end gap-2 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            {msg.role === 'assistant' && <ZAvatar />}
            <div className={`flex flex-col max-w-[75%] ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
              <div
                className={`px-3 py-2 text-sm whitespace-pre-wrap ${
                  msg.role === 'user'
                    ? 'bg-gradient-to-r from-indigo-500 to-purple-500 text-white rounded-2xl rounded-tr-sm'
                    : 'glass-card text-slate-200 rounded-2xl rounded-tl-sm'
                }`}
              >
                {msg.content}
              </div>
              <span className="text-[10px] text-slate-600 mt-1 px-1">{formatTime(msg.time)}</span>
            </div>
            {msg.role === 'user' && (
              <div className="w-7 h-7 rounded-full bg-white/10 flex items-center justify-center shrink-0">
                <User className="w-4 h-4 text-indigo-300" />
              </div>
            )}
          </motion.div>
        ))}

        {loading && (
          <div className="flex items-center gap-2">
            <ZAvatar />
            <div className="glass-card rounded-2xl rounded-tl-sm px-4 py-3 flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-slate-400 animate-bounce" />
              <span className="w-1.5 h-1.5 rounded-full bg-slate-400 animate-bounce [animation-delay:0.15s]" />
              <span className="w-1.5 h-1.5 rounded-full bg-slate-400 animate-bounce [animation-delay:0.3s]" />
            </div>
          </div>
        )}
        <div ref={endRef} />
      </div>

      <div className="p-3 border-t border-white/10 flex items-center gap-2 shrink-0">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask about your data..."
          disabled={!canChat}
          className="flex-1 bg-white/5 border border-white/10 rounded-full px-4 py-2 text-sm text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-400/50 disabled:opacity-50"
        />
        <button
          onClick={handleSend}
          disabled={loading || !input.trim()}
          className="w-9 h-9 flex items-center justify-center rounded-full bg-gradient-to-r from-indigo-500 to-purple-500 hover:shadow-[0_0_20px_rgba(139,92,246,0.7)] disabled:opacity-40 disabled:shadow-none text-white shrink-0 transition-shadow"
        >
          <Send className="w-4 h-4" />
        </button>
      </div>
    </div>
  )
}

export default ChatPanel
