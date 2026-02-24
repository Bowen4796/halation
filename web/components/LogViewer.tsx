"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import ROSLIB from "roslib";

interface RosLog {
  stamp: { sec: number; nanosec: number };
  level: number;
  name: string;
  msg: string;
  file: string;
  function: string;
  line: number;
}

interface LogEntry {
  timestamp: string;
  level: string;
  levelNum: number;
  node: string;
  message: string;
}

const LOG_LEVELS: Record<number, string> = {
  10: "DEBUG",
  20: "INFO",
  30: "WARN",
  40: "ERROR",
  50: "FATAL",
};

const CARDS = [
  { key: "DEBUG", label: "Debug", levels: [10], color: "#6b7280" },
  { key: "INFO", label: "Info", levels: [20], color: "#3b82f6" },
  { key: "WARN", label: "Warn", levels: [30], color: "#eab308" },
  { key: "ERROR", label: "Error", levels: [40, 50], color: "#ef4444" },
];

function LogCard({
  title,
  logs,
  color,
  isConnected,
}: {
  title: string;
  logs: LogEntry[];
  color: string;
  isConnected: boolean;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const isAtBottomRef = useRef(true);

  const handleScroll = () => {
    if (containerRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = containerRef.current;
      isAtBottomRef.current = scrollHeight - scrollTop - clientHeight < 30;
    }
  };

  useEffect(() => {
    if (containerRef.current && isAtBottomRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [logs]);

  return (
    <div className="bg-gray-900 rounded overflow-hidden">
      <div className="px-3 py-1.5 flex items-center justify-between border-b border-gray-800">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
          <span className="text-gray-300 text-sm font-medium">{title}</span>
        </div>
        <span className="text-gray-500 text-xs">{logs.length}</span>
      </div>
      <div
        ref={containerRef}
        onScroll={handleScroll}
        className="p-2 h-32 overflow-y-auto font-mono text-xs"
      >
        {logs.length === 0 ? (
          <div className="text-gray-600 text-center py-6">
            {isConnected ? "—" : "Disconnected"}
          </div>
        ) : (
          logs.map((log, i) => (
            <div key={i} className="text-gray-400 py-0.5">
              <span className="text-gray-600">{log.timestamp}</span>
              <span className="text-gray-500 ml-2">[{log.node}]</span>
              <span className="text-gray-300 ml-2">{log.message}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

export function LogViewer({
  ros,
  isConnected,
}: {
  ros: ROSLIB.Ros | null;
  isConnected: boolean;
}) {
  const [logs, setLogs] = useState<LogEntry[]>([]);

  const addLog = useCallback((rosLog: RosLog) => {
    const level = LOG_LEVELS[rosLog.level] || "UNKNOWN";
    const timestamp = new Date(
      rosLog.stamp.sec * 1000 + rosLog.stamp.nanosec / 1000000
    ).toLocaleTimeString();

    const entry: LogEntry = {
      timestamp,
      level,
      levelNum: rosLog.level,
      node: rosLog.name,
      message: rosLog.msg,
    };

    setLogs((prev) => [...prev.slice(-499), entry]);
  }, []);

  useEffect(() => {
    if (!ros || !isConnected) return;

    const rosoutTopic = new ROSLIB.Topic({
      ros,
      name: "/rosout",
      messageType: "rcl_interfaces/msg/Log",
    });

    rosoutTopic.subscribe((message) => {
      addLog(message as unknown as RosLog);
    });

    return () => {
      rosoutTopic.unsubscribe();
    };
  }, [ros, isConnected, addLog]);

  const getLogsForLevels = (levels: number[]) => {
    return logs.filter((log) => levels.includes(log.levelNum));
  };

  return (
    <div className="mb-6">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-lg font-semibold">Node Logs</h2>
        <button
          onClick={() => setLogs([])}
          className="text-xs text-gray-500 hover:text-gray-700"
        >
          Clear
        </button>
      </div>

      <div className="grid grid-cols-2 gap-2">
        {CARDS.map((card) => (
          <LogCard
            key={card.key}
            title={card.label}
            logs={getLogsForLevels(card.levels)}
            color={card.color}
            isConnected={isConnected}
          />
        ))}
      </div>
    </div>
  );
}
