

## 2026-09-13 08:50 ET: allocations exhausted; final evaluation recovered

Account scan shows no running/pending jobs or steps. All existing GPU allocations expired. Final report: results/generalization_20260913/final_0850.md. Tiny mixture full1000 is 483/1000=48.3%, clean54.1 delta−5.8pp CI[-8.5025,-3.1], SFT43.5 residual4.8pp CI[1.8,7.9]. Main40045.5 did not persist. Slice600/800 generated complete predictions but postprocessing assertion compared metadata limit200 with original reference limit400. CPU-only aggregate_tiny_extension.py verified exact rows, decoding fields and evaluator hash, normalized only sliced-reference dataset limit, and validated 1000 unique prompts/golds. Original failed workers retained.

T2scorefit completed actual120 and saved checkpoint120; old200=50% versus same200 SFT46.5/clean55.5. No second200: worker explicitly stopped for insufficient remaining allocation time. No main400 result. KL-only main400 remains49%, quality failure retained. SVAMP defense remains62→66→66 (no suppression), TinySVAMP greedy92vs93 stopped pre-student. RawT1/extra quality failures remain in all summaries. No universal/generalization success claim. Automatic GPU monitoring is being removed now because no allocated or queued resources remain. All experiment artifacts preserved; research objective is not declared achieved.
