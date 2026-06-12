<script lang="ts">
  // A minimal vertical bar chart. `highlight` (e.g. the predicted digit) and `truth` (the true
  // label) tint their bars so the read-out is legible at a glance.
  interface Props {
    values: number[];
    labels?: string[];
    highlight?: number | null;
    truth?: number | null;
    height?: number;
  }
  let { values, labels, highlight = null, truth = null, height = 160 }: Props = $props();

  const max = $derived(Math.max(1, ...values));
  const barColor = (i: number) => {
    if (highlight === i && truth === i) return "#7fd18a"; // correct: predicted == true
    if (highlight === i) return "#f0a35a"; // predicted
    if (truth === i) return "#5a7fd1"; // true label
    return "#3a4150";
  };
</script>

<div class="bars" style="height:{height}px">
  {#each values as v, i}
    <div class="col" title={`${labels?.[i] ?? i}: ${v}`}>
      <div class="bar-wrap">
        <span class="val">{v}</span>
        <div class="bar" style="height:{(v / max) * 100}%; background:{barColor(i)}"></div>
      </div>
      <div class="lbl" class:active={highlight === i || truth === i}>{labels?.[i] ?? i}</div>
    </div>
  {/each}
</div>

<style>
  .bars {
    display: flex;
    align-items: flex-end;
    gap: 6px;
    width: 100%;
  }
  .col {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    height: 100%;
    min-width: 0;
  }
  .bar-wrap {
    flex: 1;
    width: 100%;
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    align-items: center;
    min-height: 0;
  }
  .val {
    font-size: 10px;
    color: #8a909d;
    line-height: 1.4;
  }
  .bar {
    width: 100%;
    border-radius: 3px 3px 0 0;
    min-height: 2px;
    transition: height 0.2s ease;
  }
  .lbl {
    font-size: 11px;
    color: #7a808d;
    margin-top: 4px;
  }
  .lbl.active {
    color: #cfd6e4;
    font-weight: 600;
  }
</style>
