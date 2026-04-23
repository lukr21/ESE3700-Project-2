# SRAM read critical-path timing breakdown

Threshold convention: V_DD/2 = 0.4 V for logic; 50 mV separation for BL dev and SA internal differential. Worst read cycle picked per run.

| Stage | Signal transition | 2000 ps | 320 ps | 298 ps |
|---|---|---|---|---|
| Clock buffer (final stage) | net@30 fall → Phi2 rise | 8.1 ps | 8.1 ps | 8.1 ps |
| Clock buffer (total) | Clk fall → Phi2 rise | 42.4 ps | 40.4 ps | 40.0 ps |
| Decoder NAND | Phi2 rise → net@314 fall | 14.3 ps | 14.2 ps | 14.2 ps |
| WL buffer (INV_sized) | net@314 fall → WL1 rise | 6.8 ps | 6.8 ps | 6.8 ps |
| Decoder total | Phi2 rise → WL1 rise | 21.1 ps | 20.9 ps | 20.9 ps |
| BL development | WL1 rise → |BL−BLb|=50 mV | 2.8 ps | 2.5 ps | 2.2 ps |
| SAE AND gate | Phi2 rise → net@87 rise | 12.5 ps | 12.3 ps | 12.3 ps |
| SAE buffer tap 1 | net@87 → buf.net@0 fall | 2.9 ps | 2.9 ps | 2.9 ps |
| SAE buffer tap 2 | buf.net@0 → buf.net@1 rise | 3.2 ps | 3.2 ps | 3.2 ps |
| SAE buffer tap 3 | buf.net@1 → buf.net@6 fall | 2.7 ps | 2.7 ps | 2.7 ps |
| SAE buffer tap 4 | buf.net@6 → buf.net@2 rise | 3.1 ps | 3.1 ps | 3.1 ps |
| SAE buffer tap 5 | buf.net@2 → buf.net@9 fall | 5.4 ps | 5.4 ps | 5.4 ps |
| SAE sized INV | buf.net@9 → SAE rise | 8.6 ps | 8.5 ps | 8.5 ps |
| SAE chain total | Phi2 rise → SAE rise | 38.3 ps | 38.0 ps | 38.0 ps |
| SA differential develops | WL1 rise → |n3−n35|=50 mV | 11.4 ps | 10.2 ps | 9.6 ps |
| SA resolve to V_DD/2 | SAE rise → Q/D0 at V_DD/2 | 28.5 ps | 28.0 ps | 27.8 ps |
| Q/D bus → R latch | Q/D0 at V_DD/2 → R0 at V_DD/2 | 26.0 ps | 25.8 ps | 25.8 ps |
| Total read latency | Phi2 rise → R0 at V_DD/2 | 92.8 ps | 91.9 ps | 91.7 ps |
