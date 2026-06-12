<script lang="ts">
  // A minimal filled line chart over (x, y) points, used for spikes-over-time. Sparse x values are
  // honored (gaps in ticks render as gaps in x), so the cascade's real timing shows.
  interface Props {
    points: { x: number; y: number }[];
    height?: number;
    color?: string;
  }
  let { points, height = 180, color = "#8ab4f8" }: Props = $props();

  const W = 600; // viewBox width; SVG scales to the container via width:100%
  const H = $derived(height);
  const pad = 6;

  const geom = $derived.by(() => {
    if (points.length === 0) return { line: "", area: "", maxY: 0, maxX: 0 };
    const maxX = Math.max(1, ...points.map((p) => p.x));
    const maxY = Math.max(1, ...points.map((p) => p.y));
    const sx = (x: number) => pad + (x / maxX) * (W - 2 * pad);
    const sy = (y: number) => H - pad - (y / maxY) * (H - 2 * pad);
    const line = points.map((p, i) => `${i ? "L" : "M"}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(" ");
    const first = sx(points[0].x).toFixed(1);
    const last = sx(points[points.length - 1].x).toFixed(1);
    const area = `M${first},${(H - pad).toFixed(1)} ${line.slice(1)} L${last},${(H - pad).toFixed(1)} Z`;
    return { line, area, maxY, maxX };
  });
</script>

{#if points.length === 0}
  <div class="empty" style="height:{height}px">no somatic spikes recorded</div>
{:else}
  <div class="chart">
    <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" style="height:{height}px">
      <path d={geom.area} fill={color} fill-opacity="0.14" />
      <path d={geom.line} fill="none" stroke={color} stroke-width="1.5" vector-effect="non-scaling-stroke" />
    </svg>
    <div class="axis">
      <span>tick 0</span>
      <span class="peak">peak {geom.maxY}/tick</span>
      <span>tick {geom.maxX}</span>
    </div>
  </div>
{/if}

<style>
  .chart {
    width: 100%;
  }
  svg {
    width: 100%;
    display: block;
    background: #16181d;
    border: 1px solid #2a2d35;
    border-radius: 6px;
  }
  .axis {
    display: flex;
    justify-content: space-between;
    font-size: 10px;
    color: #6b7080;
    margin-top: 4px;
  }
  .peak {
    color: #8a909d;
  }
  .empty {
    display: flex;
    align-items: center;
    justify-content: center;
    color: #6b7080;
    font-size: 13px;
    background: #16181d;
    border: 1px solid #2a2d35;
    border-radius: 6px;
  }
</style>
