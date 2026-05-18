# Testbench Test Cases — Coverage Mapping

## TB1: tb_cp_block.v — CP Block Unit Test

**DUT:** `RTL/cp_block.v`  
**Strategy:** Set `taps_in=0` → `tree_out=0` → `biased=bias_in` directly.  
Use `nb=0` for S6–S9 tests (shifted=biased, no rounding math needed).  
Use `nb=8` only for round-half-up verification (TC02).  
Drive `a_d5`, `compute_en_d5` externally (skip cp_engine delay chain).

| TC | Mô tả | Input setup | Expected | Branch |
|----|-------|-------------|----------|--------|
| TC01 | Pipeline basic: taps=[1,2,3,4,5], w=[1,1,1,1,1], bias=0, nb=8, IN_CH=1 | tree_out=15, biased=15, shifted=0 (15+128)>>8=0 | pool_out=0 | S1-S5 pipeline |
| TC02 | Round-half-up: bias=128,nb=8 → shifted=1; bias=127,nb=8 → shifted=0 | nb=8, taps=0 | pool_out=1 then pool_out=0 | S6: round_add |
| TC03 | Clamp upper: nb=0, bias=200 → shifted=200 → clamped=127 | nb=0, taps=0, bias=200 | pool_out=127 | S7: shifted>127 |
| TC04 | Clamp lower: nb=0, bias=-200 → shifted=-200 → clamped=-127 | nb=0, taps=0, bias=-200 | pool_out=-127 | S7: shifted<-127 |
| TC05 | Clamp exact upper: nb=0, bias=127 → clamped=127 | nb=0, taps=0, bias=127 | pool_out=127 | S7: boundary |
| TC06 | Clamp exact lower: nb=0, bias=-127 → clamped=-127 | nb=0, taps=0, bias=-127 | pool_out=-127 | S7: boundary |
| TC07 | ReLU on + negative: relu_en=1, nb=0, bias=-50 → relu_out=0 | nb=0, taps=0, bias=-50, relu_en=1 | pool_out=0 | S8: relu_en&&clamped[7] |
| TC08 | ReLU on + positive: relu_en=1, nb=0, bias=50 → relu_out=50 | nb=0, taps=0, bias=50, relu_en=1 | pool_out=50 | S8: relu_en&&!clamped[7] |
| TC09 | ReLU off + negative pass-through: relu_en=0, nb=0, bias=-50 | nb=0, taps=0, bias=-50, relu_en=0 | pool_out=-50 | S8: !relu_en |
| TC10 | MaxPool — max at first: 5 pixels=[100,50,30,20,10], nb=0 | feed 5 sequential bias_in | pool_out=100 | S9: pool_cnt=0 init |
| TC11 | MaxPool — max at last: 5 pixels=[10,20,30,50,100], nb=0 | feed 5 sequential bias_in | pool_out=100 | S9: update max |
| TC12 | MaxPool — pool_write timing: fire exactly once after 5th relu_v | nb=0, 5 pixels then wait | pool_write=1 once | S9: pool_cnt==4 |
| TC13 | 2-window: 10 pixels → 2 pool_writes; verify 2nd window max | nb=0, 10 pixels=[20]*5+[5]*5 | pool_out=20 then 5 | S9: pool_cnt wraps |
| TC14 | IN_CH=4: taps=[10,0,0,0,0],w=[1,0,0,0,0],nb=0,bias=0; cycle a_d5 0..3 | acc=4×10=40 per pos | pool_out=40 | S5: acc RST/ACC |
| TC15 | IN_CH=8: same setup as TC14 | acc=8×10=80 per pos | pool_out=80 | S5: 8-cycle acc |
| TC16 | compute_en_d5=0: no out_valid, no pool_write (NOP) | ce_d5=0 held for 50 cycles | pool_write never | out_valid=0 path |
| TC17 | pool_rst: reset pool_cnt mid-window, no spurious pool_write | rst pool after 3 pixels | pool_write not before 5 new | S9: pool_rst |
| TC18 | rst: all valid registers clear, no stale pool_write | apply rst mid-pipeline | pool_write=0 after rst | rst path |

**Branch coverage:**
- S5 Accumulator `a_d5==0` (RST) vs else (ACC): TC14, TC15
- S6 `nb>0` true (nb=8): TC01, TC02; false (nb=0): TC03–TC15
- S7 Clamp `>127` / `<-127` / normal: TC03, TC04, TC05, TC06
- S8 ReLU `relu_en&&[7]` / `relu_en&&![7]` / `!relu_en`: TC07, TC08, TC09
- S9 Pool `pool_cnt==0` / `relu_out>max` / `pool_cnt==4`: TC10, TC11, TC12, TC13
- Reset paths `rst` / `pool_rst`: TC17, TC18
- `compute_en_d5==0` → `out_valid=0`: TC16

**Estimated branch coverage: ~95%**

---

## TB2: tb_layer.v — Conv1 Integration Test

**DUT:** `ecg_accelerator_top` (signals monitored internally via wire probing)  
**Scope:** Input SRAM → cp_engine (Conv1 only) → ping_pong_sram (write path)

| TC | Mô tả | Setup | Check | Module |
|----|-------|-------|-------|--------|
| TC01 | Pre-fetch: 5 shifts, compute_en=0, no pool_write | Start Conv1, watch first 25 cycles | pool_write=0 during pre-fetch | cnn_controller pre-fetch |
| TC02 | Padding out_pos=0: SRW tap[4..3]=0, tap[2]=x[0] | ECG=[10,20,30,...], Conv1 start | out_pos=0 uses padded taps | cp_engine pad_zero |
| TC03 | Normal conv out_pos=2: all 5 taps real data | same ECG | bias/rescale match hand-calc | cp_block pipeline |
| TC04 | cp_en gating: pong_we[4..7]=0 during Conv1 | cp_en=0x0F | SRAM[4..7] not written | cp_engine cp_en |
| TC05 | pong_addr tracking: 5 pool_writes → pong_addr=0..4 | Watch pong_addr reg | pong_addr increments 0..4 | cnn_controller |
| TC06 | Layer_done: after 500 pool_writes → bank_sel toggles | Run full Conv1 | bank_sel 0→1, srw_rst=1 | cnn_controller layer_done |
| TC07 | bank_sel toggle prevents read-write same bank | Conv1 done → Conv2 starts | Ping/Pong banks swap correctly | ping_pong_sram |
| TC08 | Conv1 first 3 outputs bit-exact vs Python golden (if hex available) | Load golden ECG, use weights | Compare pool_out[0..2] | end-to-end |

**Module coverage:**
- `input_sram`: write + read path
- `cp_engine`: SRW, MUX, pad_zero, weight selection
- `ping_pong_sram`: pong write, bank_sel logic
- `cnn_controller`: CONV1 counters, layer_done, pre-fetch

---

## TB3: tb_top.v — Full System Test

**DUT:** `ecg_accelerator_top`

| TC | Mô tả | Input | Check | FSM States |
|----|-------|-------|-------|------------|
| TC01 | Avalon write/read: 10 samples via DATA_IN/ADDR_IN/WR_EN | 10 writes | SRAM[0..9] readable | avalon_slave |
| TC02 | Start→busy: STATUS[0]=1 within 2 cycles after START | pulse START | busy=1 | LOAD_INPUT |
| TC03 | done_latched: STATUS[1]=1 after inference; clear on next START | full inference | done_latched toggles | DONE_S |
| TC04 | result[1:0] valid: matches Python argmax for sample 0 | golden ECG hex | result correct | all states |
| TC05 | 3 consecutive inferences: results match Python for 3 samples | 3 golden ECGs | 3 results correct | repeatability |
| TC06 | FSM coverage: all 8 states visited | TC04 run | IDLE,LOAD,CONV1-4,GAP_FC,DONE all visited | full FSM |
| TC07 | GAP sub-states: all 5 sub-states visited | TC04 run | GAP,FC,FC_FLUSH,ARGMAX,DONE_SUB | sub-FSM |

**Coverage estimate: ~90% branch**

---

## Coverage Summary

| Metric | tb_cp_block | tb_layer | tb_top | Total |
|--------|------------|---------|--------|-------|
| Branch | ~95% | ~85% | ~90% | **>90%** |
| FSM states | N/A | CONV1 subset | All 8+5 | ~100% |
| Toggle | ~90% | ~85% | ~90% | ~90% |

**Pass criteria:**
- tb_cp_block: all 18 TC PASS with exact values
- tb_layer: Conv1 positions 0,1,2 match hand-calc; layer_done at correct cycle
- tb_top: result[1:0] matches Python argmax for ≥3 test samples
