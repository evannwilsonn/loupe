# Loupe demo: design notes

The page is **a mail-sorting cabinet**. Photos are posts, the network is the clerk, and the ten classes are ten wooden pigeonholes with brass label holders.

- **Out tray:** a dark oak tray holding twenty unsorted test photos, plus a brass "Post your own photo" button.
- **Reading slip:** a perforated paper slip. It shows the photo enlarged, with every pixel as a square, and the clerk's top five readings as a printed tally.
- **Cabinet:** two rows of five pigeonholes.
  - Each one holds the test photos the model filed there.
  - A photo in the wrong box gets a red tag naming what it really is.
  - Each brass label gives the box's purity on the full test set: the share of photos filed there that belong there, i.e. the confusion-matrix column.
  - When you post a photo, it flies from the slip into its box, and the box lights up.
- **The day's tally:** each method's held-out accuracy on mechanical counter wheels. The wheels roll into place once, when scrolled into view (not with reduced motion), and each counter shows its 95% interval. The final model's counter has a red frame.
- **Learning curves:** accuracy on held-out photos (solid) and on training photos (dashed) per epoch. Orange is without augmentation, blue is with it.
- **Misfiled most often:** the top confusions as a ledger list ("99 cats filed under dog"). The full grid is in a `<details>` table.

## Tokens (defined at the top of `template.html`)

- **Office hours (light theme):**
  - Plaster wall `#e9e1d0` and oak `#a8743f`, with darker oak frames.
  - Near-black pigeonhole interiors `#3a2614`.
  - Brass labels `#c9a24a` and paper slips `#f8f3e6`.
  - Tag red `#c22f1f` is the one accent.
- **After hours (dark theme):** dark walnut and a dim wall, with its own steps rather than an inverted light theme.
- **Chart colours:** orange `#c2410c` and blue `#1769aa` (light), `#e0662d` and `#4b97de` (dark). All four pass the dataviz palette check against the slip colour, and line dashes separate training from held-out accuracy.
- **Type:**
  - Rokkitt, a slab serif, for headings and labels: stencilled, office-furniture lettering.
  - Overpass for reading.
  - Overpass Mono for numbers, slips and counters.

## Things to avoid

- A third accent colour.
- Generic rounded cards.
- Gradients other than the slip's perforated edge.
- Making the cabinet a plain image grid. The boxes, brass labels and red tags are the point.
