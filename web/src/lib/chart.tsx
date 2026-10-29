import { BarChart, LineChart, ScatterChart } from "echarts/charts";
import {
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  MarkAreaComponent,
  MarkLineComponent,
  TooltipComponent,
} from "echarts/components";
import * as echarts from "echarts/core";
import type { EChartsCoreOption } from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";
import { useEffect, useRef } from "react";
import { PALETTE } from "./states";

echarts.use([
  LineChart,
  BarChart,
  ScatterChart,
  GridComponent,
  TooltipComponent,
  LegendComponent,
  MarkLineComponent,
  MarkAreaComponent,
  DataZoomComponent,
  CanvasRenderer,
]);

const FONT = "IBM Plex Sans, system-ui, sans-serif";

/** Réglages communs : axes discrets, grille légère, police de l'interface. */
export function baseOption() {
  return {
    textStyle: { fontFamily: FONT, color: PALETTE.inkSoft, fontSize: 12 },
    animationDuration: 200,
    grid: { left: 52, right: 16, top: 28, bottom: 32, containLabel: false },
    tooltip: {
      trigger: "axis",
      backgroundColor: "#ffffff",
      borderColor: PALETTE.line,
      borderWidth: 1,
      textStyle: { color: PALETTE.ink, fontFamily: FONT, fontSize: 13 },
      extraCssText: "box-shadow:none;border-radius:6px;",
    },
    legend: {
      top: 0,
      left: 0,
      itemWidth: 14,
      itemHeight: 8,
      textStyle: { color: PALETTE.inkSoft, fontFamily: FONT, fontSize: 12 },
    },
  };
}

export const axisStyle = {
  axisLine: { lineStyle: { color: PALETTE.line } },
  axisTick: { show: false },
  axisLabel: { color: PALETTE.inkSoft, fontSize: 12 },
  splitLine: { lineStyle: { color: "#f1eef5" } },
};

type Props = { option: EChartsCoreOption; height?: number; className?: string; ariaLabel: string };

export function Chart({ option, height = 260, className, ariaLabel }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const chart = useRef<echarts.ECharts | null>(null);

  useEffect(() => {
    if (!ref.current) return;
    chart.current = echarts.init(ref.current, undefined, { renderer: "canvas" });
    const observer = new ResizeObserver(() => chart.current?.resize());
    observer.observe(ref.current);
    return () => {
      observer.disconnect();
      chart.current?.dispose();
      chart.current = null;
    };
  }, []);

  useEffect(() => {
    chart.current?.setOption(option, { notMerge: true });
  }, [option]);

  return <div ref={ref} role="img" aria-label={ariaLabel} className={className} style={{ height }} />;
}
