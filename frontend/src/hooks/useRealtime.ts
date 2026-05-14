/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See LICENSE in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

import { useCallback, useEffect, useRef, useState } from 'react'
import { Message } from '../types'

interface RealtimeOptions {
  agentId?: string | null
  onMessage?: (msg: any) => void
  onAudioDelta?: (delta: string) => void
  onTranscript?: (role: 'user' | 'assistant', text: string) => void
}

export function useRealtime(options: RealtimeOptions) {
  const [connected, setConnected] = useState(false)
  const [messages, setMessages] = useState<Message[]>([])
  const wsRef = useRef<WebSocket | null>(null)
  const audioRecording = useRef<any[]>([])
  const conversationRecording = useRef<any[]>([])

  const connect = useCallback(async () => {
    const config = await fetch('/api/config').then(r => r.json())
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:'
    const ws = new WebSocket(
      `${protocol}//${location.host}${config.ws_endpoint}`
    )

    ws.onopen = () => {
      setConnected(true)
      if (options.agentId) {
        // 포인트:
        // WebSocket이 열리면 먼저 사용할 Agent ID를 보냅니다.
        // 서버는 이 ID로 시나리오 지시문과 모델 설정을 찾아 Voice Live 세션에 연결합니다.
        ws.send(
          JSON.stringify({
            type: 'session.update',
            session: { agent_id: options.agentId },
          })
        )
      }
    }

    ws.onmessage = event => {
      const msg = JSON.parse(event.data)
      options.onMessage?.(msg)

      switch (msg.type) {
        case 'response.audio.delta':
          if (msg.delta) {
            // AI 답변 음성이 완성될 때까지 기다리지 않고, 생성되는 오디오 조각을 즉시 재생합니다.
            options.onAudioDelta?.(msg.delta)
            audioRecording.current.push({
              type: 'assistant',
              data: msg.delta,
              timestamp: new Date().toISOString(),
            })
          }
          break
        case 'conversation.item.input_audio_transcription.completed':
          if (msg.transcript) {
            // 사용자의 음성이 Speech-to-Text로 확정되면 채팅 패널에 사용자 문장으로 표시합니다.
            const message: Message = {
              id: crypto.randomUUID(),
              role: 'user',
              content: msg.transcript,
              timestamp: new Date(),
            }
            setMessages(prev => [...prev, message])
            conversationRecording.current.push({
              role: 'user',
              content: msg.transcript,
            })
            options.onTranscript?.('user', msg.transcript)
          }
          break
        case 'response.audio_transcript.done':
          if (msg.transcript) {
            // LLM이 만든 답변은 Text-to-Speech로 나가면서 동시에 텍스트 transcript도 받습니다.
            const message: Message = {
              id: crypto.randomUUID(),
              role: 'assistant',
              content: msg.transcript,
              timestamp: new Date(),
            }
            setMessages(prev => [...prev, message])
            conversationRecording.current.push({
              role: 'assistant',
              content: msg.transcript,
            })
            options.onTranscript?.('assistant', msg.transcript)
          }
          break
      }
    }

    ws.onclose = () => {
      setConnected(false)
    }
    wsRef.current = ws
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [options.agentId])

  const send = useCallback((data: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(typeof data === 'string' ? data : JSON.stringify(data))
    }
  }, [])

  const clearMessages = useCallback(() => {
    setMessages([])
    conversationRecording.current = []
    audioRecording.current = []
  }, [])

  const getRecordings = useCallback(
    () => ({
      conversation: conversationRecording.current,
      audio: audioRecording.current,
    }),
    []
  )

  useEffect(() => {
    if (!options.agentId) return
    connect()
    return () => wsRef.current?.close()
  }, [connect, options.agentId])

  return {
    connected,
    messages,
    send,
    clearMessages,
    getRecordings,
  }
}
