# Loupe demo: design notes

The page is **a mail-sorting cabinet**. Photos are posts, the network is the clerk, and the ten classes are ten steel pigeonholes with yellow label holders.

- **Out tray:** a cobalt tray holding twenty unsorted test photos, plus a yellow "Post your own photo" button.
- **Reading slip:** a perforated paper slip. It shows the photo enlarged, with every pixel as a square, and the clerk's top five readings as a printed tally.
- **Cabinet:** two rows of five pigeonholes.
  - Each one holds the test photos the model filed there.
  - A photo in the wrong box gets a red tag naming what it really is.
  - Each yellow label gives the box's purity on the full test set: the share of photos filed there that belong there, i.e. the confusion-matrix column.
  - When you post a photo, it flies from the slip into its box, and the box lights up.
- **The day's tally:** each method's held-out accuracy on mechanical counter wheels. The wheels roll into place once, when scrolled into view (not with reduced motion), and each counter shows its 95% interval. The final model's counter has a red frame.
- **Learning curves:** accuracy on held-out photos (solid) and on training photos (dashed) per epoch. Orange is without augmentation, blue is with it.
- **Misfiled most often:** the top confusions as a ledger list ("99 cats filed under dog"). The full grid is in a `<details>` table.

## Tokens (defined at the top of `template.html`)

- **Office hours (light theme):** a cobalt-blue enamelled steel cabinet on a pale cool-grey wall.
  - Wall `#e8ebef`, cabinet `#2448a8` with darker frames, near-black navy pigeonhole interiors `#0d1730`.
  - Safety-yellow label holders `#ffd23f` with black lettering.
  - White slips, and a red-orange tag `#e8401c` as the one accent.
  - The palette is cool on purpose, unlike the warm cream, brown and red of the earlier projects.
- **After hours (dark theme):** a near-black blue room `#0b0f18` with a deep cobalt cabinet and a slightly dimmer yellow, on its own steps rather than an inverted light theme.
- **Chart colours:** orange `#d9480f` and cobalt `#2448a8` (light), `#e0662d` and `#5b8def` (dark). All four pass the dataviz palette check against both the wall and the slip colour, and line dashes separate training from held-out accuracy.
- **Type:**
  - Rokkitt, a slab serif, for headings and labels: stencilled, office-furniture lettering.
  - Overpass for reading.
  - Overpass Mono for numbers, slips and counters.

## Things to avoid

- A third accent colour.
- Generic rounded cards.
- Gradients other than the slip's perforated edge.
- Making the cabinet a plain image grid. The boxes, yellow labels and red tags are the point.
