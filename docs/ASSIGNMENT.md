# Harvix take-home assignment

## Background

Harvix builds robots that harvest tomatoes in commercial greenhouses. Real labelled data from inside a greenhouse is slow and expensive to collect, so we want to bootstrap our models with synthetic data.

The `assets/` folder in this package holds our 3D assets: the robot and a full glasshouse with crop. `assets/ASSETS.md` describes what each file is.

## The task

**Build a synthetic dataset we could use to train the models that let our robot harvest tomatoes.**

That's the whole brief, and it is open on purpose. You decide what goes in the dataset and why. Make the calls an engineer on our team would have to make, write down your assumptions, and build it.

## What to hand in

1. **The dataset.** Either the full dataset or a representative sample plus a way to generate more. Large files can go on a link (Drive, S3, etc.).
2. **The code.** Someone else on the team should be able to regenerate your data from a clean checkout. Tell us how.
3. **A short write-up** of at most 2 pages, covering:
   - what you built and who it is for,
   - the decisions and trade-offs you made, and what you deliberately left out,
   - how you would know the data is any good,
   - what you would do with another month.
4. **Your AI usage, unedited.** We expect you to use AI tools and we want to see how you work with them. Export every chat, agent session and prompt history you used (Claude, ChatGPT, Cursor, Copilot Chat, whatever), including the dead ends. Don't tidy them up. We are judging how you work, not whether you used AI.

## Practicalities

- **Time:** you have one week. Plan for about 15 hours of work, and tell us roughly how long you actually spent. We would rather see a smaller piece of work with clear reasoning than a large one we can't follow.
- **Hardware:** use what you have. If compute limited you, say how and what you did about it. We judge your decisions, not your frame count.
- **Software:** the `.blend` files need Blender 5.2 or newer (free from blender.org). Otherwise use any tools, languages and libraries you like.
- **The robot:** treat the robot's geometry and mechanisms as fixed, because they reflect the real machine. Everything else is yours to change.
- **Questions:** you can email us, and we'll answer the way a stakeholder would. Writing down an assumption and moving on is also fine.

## Submitting

Send a link to a git repository or an archive containing the code, the write-up, the AI logs, and the dataset or a link to it.

Good luck, and have fun with it.
