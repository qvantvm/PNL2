# Evaluation Report

## Baselines
- **learned**: mean P@10=0.140, MRR=0.600
- **random**: mean P@10=0.020, MRR=0.022
- **feature**: mean P@10=0.360, MRR=0.867
- **untrained**: mean P@10=0.020, MRR=0.022

## Intrinsic nearest neighbors
- DURATION:1/4: P@k=0.000 MRR=0.000 → ['OCTAVE:5', 'ACCIDENTAL:NATURAL', 'EVENT:NOTE', 'PITCH_CLASS:D', 'PITCH_CLASS:G']
- PITCH_CLASS:C: P@k=0.300 MRR=1.000 → ['PITCH_CLASS:E', 'PITCH_CLASS:D', 'FINGER:4', 'FINGER:5', 'ACCIDENTAL:NATURAL']
- PEDAL:DOWN: P@k=0.100 MRR=1.000 → ['PEDAL:UP', 'TEMPO_BPM:112', 'TEMPO_BPM:116', 'TEMPO_BPM:82', 'TEMPO_BPM:74']
- DYNAMIC:MF: P@k=0.300 MRR=1.000 → ['DYNAMIC:PP', 'ARTICULATION:TENUTO', 'DYNAMIC:FF', 'PITCH_CLASS:A', 'ARTICULATION:ACCENT']
- ARTICULATION:STACCATO: P@k=0.000 MRR=0.000 → ['DURATION:1/2', 'OCTAVE:4', 'REST', 'TIE:END', 'TIE:START']

## Notes
- Intrinsic expected sets are soft musical heuristics; not all relationships must emerge from every corpus.
- Namespace probes can be partly trivial because labels are encoded in token identity.
