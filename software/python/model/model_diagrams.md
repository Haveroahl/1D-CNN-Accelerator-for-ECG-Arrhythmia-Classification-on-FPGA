# ECG 1D-CNN Model Diagrams

Source files:

- `software/python/model/model.py`: `ECG_1DCNN`
- `software/python/prune_finetune.py`: `ECG_1DCNN_Pruned`
- `software/python/results/v3_results_pruned.json`: saved pruning metadata

## Dense Model: `ECG_1DCNN`

```mermaid
flowchart LR
    X["Input ECG\n(B, 1, 2500)"]
    C1["Conv1d\n1 -> 4, K=5, P=2\nbias=True\n(B, 4, 2500)"]
    P1["MaxPool1d\nK=5\n(B, 4, 500)"]
    C2["Conv1d\n4 -> 8, K=5, P=2\nbias=True\n(B, 8, 500)"]
    P2["MaxPool1d\nK=5\n(B, 8, 100)"]
    C3["Conv1d\n8 -> 8, K=5, P=2\nbias=True\n(B, 8, 100)"]
    P3["MaxPool1d\nK=5\n(B, 8, 20)"]
    C4["Conv1d\n8 -> 16, K=5, P=2\nbias=True\n(B, 16, 20)"]
    R4["ReLU\nonly after Conv4"]
    P4["MaxPool1d\nK=5\n(B, 16, 4)"]
    GAP["AdaptiveAvgPool1d(1)\n(B, 16, 1)"]
    S["squeeze(-1)\n(B, 16)"]
    FC["Linear\n16 -> 4\n(B, 4 logits)"]
    Y["Classes\nAFIB / GSVT / SB / SR"]

    X --> C1 --> P1 --> C2 --> P2 --> C3 --> P3 --> C4 --> R4 --> P4 --> GAP --> S --> FC --> Y
```

Dense parameter count: `1244`.

| Layer | Channels | Output shape | Parameters |
|---|---:|---:|---:|
| Input | 1 | `(B, 1, 2500)` | - |
| Conv1 + Pool | `1 -> 4` | `(B, 4, 500)` | 24 |
| Conv2 + Pool | `4 -> 8` | `(B, 8, 100)` | 168 |
| Conv3 + Pool | `8 -> 8` | `(B, 8, 20)` | 328 |
| Conv4 + ReLU + Pool | `8 -> 16` | `(B, 16, 4)` | 656 |
| GAP | - | `(B, 16)` | - |
| FC | `16 -> 4` | `(B, 4)` | 68 |

## Pruned Model: `ECG_1DCNN_Pruned`

```mermaid
flowchart LR
    X["Input ECG\n(B, 1, 2500)"]
    C1["Conv1d\n1 -> 4, K=5, P=2\nbias=True\n(B, 4, 2500)"]
    P1["MaxPool1d\nK=5\n(B, 4, 500)"]
    C2["Conv1d\n4 -> 4, K=5, P=2\nbias=True\n(B, 4, 500)"]
    P2["MaxPool1d\nK=5\n(B, 4, 100)"]
    C3["Conv1d\n4 -> 8, K=5, P=2\nbias=True\n(B, 8, 100)"]
    P3["MaxPool1d\nK=5\n(B, 8, 20)"]
    C4["Conv1d\n8 -> 8, K=5, P=2\nbias=True\n(B, 8, 20)"]
    R4["ReLU\nonly after Conv4"]
    P4["MaxPool1d\nK=5\n(B, 8, 4)"]
    GAP["AdaptiveAvgPool1d(1)\n(B, 8, 1)"]
    S["squeeze(-1)\n(B, 8)"]
    FC["Linear\n8 -> 4\n(B, 4 logits)"]
    Y["Classes\nAFIB / GSVT / SB / SR"]

    X --> C1 --> P1 --> C2 --> P2 --> C3 --> P3 --> C4 --> R4 --> P4 --> GAP --> S --> FC --> Y
```

Pruned parameter count: `640`, reduction `48.55%`.

| Layer | Channels | Output shape | Parameters |
|---|---:|---:|---:|
| Input | 1 | `(B, 1, 2500)` | - |
| Conv1 + Pool | `1 -> 4` | `(B, 4, 500)` | 24 |
| Conv2 + Pool | `4 -> 4` | `(B, 4, 100)` | 84 |
| Conv3 + Pool | `4 -> 8` | `(B, 8, 20)` | 168 |
| Conv4 + ReLU + Pool | `8 -> 8` | `(B, 8, 4)` | 328 |
| GAP | - | `(B, 8)` | - |
| FC | `8 -> 4` | `(B, 4)` | 36 |

## Structured Channel Pruning

```mermaid
flowchart TB
    F["Trained dense checkpoint\nbest_model.pth"]
    R["Rank conv output channels\nTaylor score with train_loader\nfallback: L1-norm"]
    K["Keep selected channels\nConv1: 0,1,2,3\nConv2: 0,1,3,5\nConv3: 0,1,2,3,4,5,6,7\nConv4: 0,1,2,3,10,13,14,15"]
    T["Transfer weights\nConv input channels follow previous kept outputs\nFC input follows kept Conv4 outputs"]
    FT["Fine-tune pruned model\nPhase 1: 30 epochs @ 1e-3\nPhase 2: 20 epochs @ 1e-4"]
    O["Saved hardware target\nbest_model_pruned.pth\nchannels 1 -> 4 -> 4 -> 8 -> 8\nFC 8 -> 4"]

    F --> R --> K --> T --> FT --> O
```

## Dense vs Pruned

```mermaid
flowchart LR
    subgraph D["Dense ECG_1DCNN"]
        D0["Input 1 x 2500"]
        D1["Conv1 1 -> 4"]
        D2["Conv2 4 -> 8"]
        D3["Conv3 8 -> 8"]
        D4["Conv4 8 -> 16"]
        D5["GAP + FC 16 -> 4"]
        D0 --> D1 --> D2 --> D3 --> D4 --> D5
    end

    subgraph P["Pruned ECG_1DCNN_Pruned"]
        P0["Input 1 x 2500"]
        P1["Conv1 1 -> 4"]
        P2["Conv2 4 -> 4"]
        P3["Conv3 4 -> 8"]
        P4["Conv4 8 -> 8"]
        P5["GAP + FC 8 -> 4"]
        P0 --> P1 --> P2 --> P3 --> P4 --> P5
    end

    D2 -. "prune 4 output channels" .-> P2
    D4 -. "prune 8 output channels" .-> P4
    D5 -. "FC input shrinks 16 -> 8" .-> P5
```
