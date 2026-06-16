# Warehouse Colour-Sort Challenge

Train a Franka Panda arm to pick parcels from the inbound zone and place each into the output bin
that matches the colour of the **tag on top of the parcel**. Your score is the number of parcels
sorted into the correct bin.

You are given the environment, observations, actions, and success condition. **Reward shaping is up
to you — that's the core of your submission.** A sparse reward is the default, and a simple worked
example (state-based) is included to learn from; the teams that design the best reward (and policy)
win.

---

## The task

- A Franka Panda arm with a parallel gripper is mounted at a tabletop workstation.
- **Parcels** look like brown warehouse cardboard boxes. Each carries a **coloured rectangular tag**
  (a sticker/label) on its **top face** — red or blue. The tag, not the box
  colour, tells you where the parcel goes.
- Parcels spawn in the **inbound zone** in the centre of the table, in front of the robot.
- There is one **colour-coded output bin per tag colour** — a red bin, a blue bin, etc. Bins are
  **low-walled and wide**, placed to the **left and right** of the robot.
- **Goal:** place each parcel into the bin whose colour matches the parcel's tag.
- **Score:** number of correctly-sorted parcels per episode.

### Colour → bin mapping

Tag colour → matching bin colour (red tag → red bin, blue tag → blue bin). The mapping is by **bin
colour, not by side** — at hard difficulty the bins' left/right positions can swap between episodes,
so you must identify the bins by colour, not location.

---

## Difficulty levels

Switch with `difficulty=easy|medium|hard`.

| Level  | Parcels | Layout                 | Tag colours | Observation default |
|--------|---------|------------------------|-------------|---------------------|
| easy   | 2       | fixed poses            | 2           | **state** (plumbing check) |
| medium | 4–6     | randomised poses, tag always top-visible | 2 | **rgb** (wrist cam) |
| hard   | 6–8     | randomised + light clutter, tag always top-visible | 2 | **rgb** (wrist cam) + heavy randomisation |

**Easy** exists only to confirm everything runs — it hands you tag colours directly as state.
**Medium and hard are the real challenge** and are vision-based: your policy sees the world through
the Panda's wrist camera and must detect each parcel's tag and the bin colours.

**Hard adds, on top of medium:**
- **Bin positions swap** between episodes (the red bin may be on the left or the right) — you must
  identify bins by colour, not memorise a side.
- **Background variation:** table and floor colours change.
- **Lighting variation.**
- **Appearance variation:** slightly different cardboard shades and tag-colour shades.
- A **secondary speed metric** (how fast you sort), used as a tiebreaker — correct-sort count is
  still the primary score.

---