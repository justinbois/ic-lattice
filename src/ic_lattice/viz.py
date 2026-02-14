import bokeh.layouts
import bokeh.models
import bokeh.palettes
import bokeh.plotting

def display(lattice, frame_height=400):
    """Display an interactive plot of lattice populated with A (yellow)
    and B (purple) molecules.

    Parameters
    ----------
    lattice : list of 2D Numpy arrays
        Lattices to visualize

    Returns
    -------
    output : Bokeh layout
        Plot with a slider to go through simulation steps.
    """
    if not isinstance(lattice, list):
        lattice = [lattice]

    # Get shape of domain
    n, m = lattice[0].shape

    # Set up figure with appropriate dimensions
    frame_width = int(m / n * frame_height)

    # Build the plot
    p = bokeh.plotting.figure(
        frame_height=frame_height,
        frame_width=frame_width,
        x_range=[0, m],
        y_range=[0, n],
    )

    # Build the images to display
    ims = [lat for lat in lattice]
    color = bokeh.models.LinearColorMapper(
        bokeh.palettes.HighContrast[3][:3:2], low=0, high=1
    )

    cds = bokeh.models.ColumnDataSource(dict(image=[ims[0]]))
    p.image(image="image", x=0, y=0, dw=m, dh=n, source=cds, color_mapper=color)

    slider = bokeh.models.Slider(
        start=0,
        end=len(lattice) - 1,
        value=0,
        step=1,
        title="MC iteration",
    )

    callback = bokeh.models.CustomJS(
        args=dict(cds=cds, slider=slider, ims=ims, m=m, n=n),
        code="""
    cds.data['image'] = [ims[slider.value]];
    cds.change.emit();
    """,
    )

    slider.js_on_change("value", callback)

    layout = bokeh.layouts.column(
        bokeh.layouts.row(bokeh.layouts.Spacer(width=10), slider), p
    )

    return layout

