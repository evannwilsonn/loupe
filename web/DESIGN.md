# Loupe demo: design notes

The page is **a darkroom light table**. Test photos lie on it as a contact sheet of film strips, marked up with a red grease pencil the way a photographer marks a sheet:
- a **circle** means the network got the photo right;
- a **cross**, with its wrong guess written under the frame, means it didn't.

A **loupe**, a round magnifier with a dark rim, shows any frame enlarged with every pixel visible, next to the network's reading of it.

## Layout

- **Film edge:** a black band across the top with sprocket holes and amber edge print, which carries the navigation.
- **Intro:** a large serif question ("What's in the frame?") and three grease-pencil figures.
- **Bench:**
  - On the left, the contact sheet: ten strips, one per class, with twelve frames each.
  - On the right, the loupe, sticky on wide screens.
  - Below 1000px the loupe sits under the sheet. On phones each strip scrolls sideways inside itself, and the page never scrolls sideways.
- **Test strip:** the accuracy ladder as horizontal bands, one per method, with a thin line for the 95% interval. The final model's band is grease red.
- **Learning curves:** accuracy on held-out photos (solid) and on training photos (dashed) per epoch. Orange is training without augmentation, blue is with it.
- **Proof sheet:** the confusion matrix as a grid of cells in one hue. Mistakes are shaded four times darker so they show, and the text says so.
- **Footer:** another strip of film edge.

## Tokens (defined at the top of `template.html`)

- **Room light (light theme):** a cool grey-white table (`#eef0ec`), near-black film (`#17181a`), amber edge print (`#e3a33b`) and grease red (`#d0261c`).
- **Safelight (dark theme):** a warm black table under a red safelight (`#17110f`), with rose-cream text and a brighter grease red. It has its own steps, not an inverted light theme.
- **Chart colours:**
  - Room light: orange `#c2410c` and blue `#1769aa`.
  - Safelight: orange `#e0662d` and blue `#4b97de`.
  - All four pass the dataviz palette check against their background, and the training and held-out lines also differ in dash pattern.
- **Type:**
  - Gloock for display.
  - Hanken Grotesk for reading.
  - Share Tech Mono for edge print, numbers and charts.
  - Permanent Marker only for grease-pencil writing.

## Things to avoid

- Card shadows, except the loupe's lens.
- Gradients.
- A third accent colour.
- Turning the contact sheet into a generic image grid with rounded cards.
- Any film manufacturer's name or edge print.
