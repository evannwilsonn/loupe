# Loupe demo: design notes

The page shares its look with the project thumbnail (`.github/thumbnail.jpg`): a contact sheet of tiny photos, and a jeweller's loupe that enlarges one of them until its pixels show. The ten classes are still a sorting cabinet with yellow label tags.

- **Masthead:** the 120 test photos drawn as a dim mosaic behind the page (a canvas, redrawn on resize), veiled on the left like the thumbnail. On top: LOUPE in Tektur with wide tracking, a short yellow bar, the headline in Jura, and the big test-accuracy figure.
- **Out tray:** a near-black contact sheet with sprocket rules, holding twenty unsorted test photos, plus a yellow "Post your own photo" button.
- **The loupe:** the photo being read sits inside a round lens with a pale rim, a cobalt ring and a cobalt handle, with a 32×32 pixel grid over it. A yellow label tag above the lens gives the first guess and its probability ("ship 0.94"). The top five readings print below.
- **Cabinet:** a cobalt frame holding ten dark trays.
  - Each one holds the test photos the model filed there.
  - A photo in the wrong box gets a red tag naming what it really is.
  - Each yellow label gives the box's purity on the full test set: the share of photos filed there that belong there, i.e. the confusion-matrix column.
  - When you post a photo, it flies from the lens into its box, and the box lights up.
- **The day's tally:** each method's held-out accuracy on counter wheels set in Tektur. The wheels roll into place once, when scrolled into view (not with reduced motion). The final model's counter is a dark plate with yellow digits and a yellow frame, like the thumbnail's figure.
- **Learning curves** and **Misfiled most often** are unchanged in content.

## Tokens (defined at the top of `template.html`)

- **After hours (dark theme)** is the thumbnail: ground `#0d1117`, pale rim and text `#dfe6f3`, cobalt `#2f5fe0` (with `#7d9dff` for cobalt text, which needs more contrast), safety yellow `#ffd23f` for tags, bars and the big figure.
- **Office hours (light theme):** the same loupe on a pale cool lightbox `#eef1f6` with white slips and dark ink `#0d1117`. Cobalt text darkens to `#2448b8`. Yellow is used only as a fill (tags, bars, a highlighter band behind the big figure), never as text on the pale ground. The tray, cabinet trays and the final counter stay dark in both themes, because photos read best on black.
- **Chart colours:** orange `#d9480f` and cobalt `#2f5fe0` (light), `#e0662d` and `#6f93ff` (dark); line dashes separate training from held-out accuracy.
- **Type:** Tektur (headings, wordmark, labels, counters) and Jura (reading, slips, tables, with tabular figures), self-hosted from `docs/fonts/` under the SIL Open Font License.

## Things to avoid

- A third accent colour beyond cobalt and yellow (the red tag is a signal, not decoration).
- Rounded cards and soft shadows; edges are square, rules are thin.
- Making the cabinet a plain image grid. The boxes, yellow labels and red tags are the point.
