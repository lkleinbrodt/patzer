For v4, we implemented the WSD schedule.

See models.md and project_log.md for details.

We trained v4 with a warmup and constant learning rate for ~145k iterations and then it stopped early due to early stopping.

Then i went in and manually created a new config file for the cooldown phase train_patzer_v4_cooldown.py, and restarted the training from the last available checkpoint.

the alst available checkpoint was at iteration 140000.

almost immediately we see val loss drop by a lot.

over the next few thousand iterations we saw immense improvements in val loss, dropping more quickly  in a few iterations than in tens of thousands of iterations with the constant learning rate.

i'll provide you the loss logs. it's a little messy because we trained iters 140-145k twice. the first time you see them was with the original constant learning rate, and then we restarted training from 140k so we trained those iterations again but this time with the new cooldown schedule.

so you can see how much faster loss dropped with the new cooldown schedule.

Anyways here's my question:

We saw such immense gains with the smaller learning rate, and now i'm wondering if we left gains on the table by having the constant learning rate for so long.

Should we adjust our approach? should we change our learning schedule? go back to cosine? should i retrain v4 with a different learning rate schedule?

Introspect on this for a bit, taking into account model architecture, previous training run learnings (from models.md and project_log.md), and the fact that we saw immense gains with the smaller learning rate.

(note a similar blip happened at 160k-161k iterations where i had to restart training again).