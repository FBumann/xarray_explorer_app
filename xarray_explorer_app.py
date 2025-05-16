import functools
import inspect
import io
import os
import sys
import tempfile
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union, cast

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import xarray as xr


T = TypeVar('T')


def show_traceback(
        return_original_input: bool = False, include_args: bool = True, container: Optional[Any] = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    A decorator that shows the full traceback in Streamlit when an exception occurs.

    Args:
        return_original_input: If True and the first argument is not None, return it on error.
                              Useful for data processing functions to return original data.
        include_args: If True, show function arguments in the error details.
        container: Optional Streamlit container to display errors in.
                  If None, uses st directly.

    Returns:
        The decorated function
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get display container
            display = container if container is not None else st

            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Show error message
                display.error(f'⚠️ Error in {func.__name__}: {str(e)}')

                # Create an expander for detailed error info
                with display.expander('See detailed traceback'):
                    # Show the full traceback
                    display.code(traceback.format_exc(), language='python')

                    # Show function info if requested
                    if include_args:
                        display.markdown('**Function Information:**')

                        # Try to get source code
                        try:
                            display.code(inspect.getsource(func), language='python')
                        except Exception:
                            display.warning('Could not retrieve function source code.')

                        # Show arguments
                        display.markdown('**Function Arguments:**')

                        # Safely represent args
                        safe_args = []
                        for arg in args:
                            try:
                                repr_arg = repr(arg)
                                if len(repr_arg) > 200:  # Truncate long representations
                                    repr_arg = repr_arg[:200] + '...'
                                safe_args.append(repr_arg)
                            except Exception:
                                safe_args.append('[Representation failed]')

                        # Safely represent kwargs
                        safe_kwargs = {}
                        for k, v in kwargs.items():
                            try:
                                repr_v = repr(v)
                                if len(repr_v) > 200:  # Truncate long representations
                                    repr_v = repr_v[:200] + '...'
                                safe_kwargs[k] = repr_v
                            except Exception:
                                safe_kwargs[k] = '[Representation failed]'

                        # Display args and kwargs
                        display.text(f'Args: {safe_args}')
                        display.text(f'Kwargs: {safe_kwargs}')

                # Also log to console/stderr for server logs
                print(f'Exception in {func.__name__}:', file=sys.stderr)
                traceback.print_exc(file=sys.stderr)

                # Determine what to return on error
                if return_original_input and args and args[0] is not None:
                    # Return the first argument (usually the data being processed)
                    return args[0]
                else:
                    # Return None as default
                    return None

        return cast(Callable[..., T], wrapper)

    return decorator


@show_traceback()
def download_data(filtered_data: xr.DataArray, var_name: str, download_format: str, container: Any) -> None:
    """Creates download buttons for the filtered data.

    Args:
        filtered_data: The filtered data to download.
        var_name: Name of the variable.
        download_format: Format to download (CSV, NetCDF, Excel).
        container: Streamlit container to place the download button.
    """
    if download_format == 'CSV':
        csv = filtered_data.to_dataframe().reset_index().to_csv(index=False)
        container.download_button(label='Download CSV', data=csv, file_name=f'{var_name}_filtered.csv', mime='text/csv')
    elif download_format == 'NetCDF':
        # Create temp file for netCDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nc') as tmp:
            filtered_data.to_netcdf(tmp.name)
            with open(tmp.name, 'rb') as f:
                container.download_button(
                    label='Download NetCDF',
                    data=f.read(),
                    file_name=f'{var_name}_filtered.nc',
                    mime='application/x-netcdf',
                )
    elif download_format == 'Excel':
        # Create in-memory Excel file
        buffer = io.BytesIO()
        filtered_data.to_dataframe().reset_index().to_excel(buffer, index=False)
        buffer.seek(0)

        container.download_button(
            label='Download Excel',
            data=buffer,
            file_name=f'{var_name}_filtered.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        )


@show_traceback()
def display_data_info(data: Union[xr.Dataset, xr.DataArray], container: Optional[Any] = None) -> None:
    """
    Display basic information about an xarray object.

    Args:
        data: xarray.Dataset or xarray.DataArray
        container: Streamlit container to render in (if None, uses st directly)
    """
    if container is None:
        container = st

    # Show dimensions and their sizes
    container.write('**Dimensions:**')
    dim_df = pd.DataFrame({'Dimension': list(data.sizes.keys()), 'Size': list(data.sizes.values())})
    container.dataframe(dim_df)

    # For Dataset, show variables
    if isinstance(data, xr.Dataset):
        container.write('**Variables:**')
        var_info = []
        for var_name, var in data.variables.items():
            var_info.append({'Variable': var_name, 'Dimensions': ', '.join(var.dims), 'Type': str(var.dtype)})
        container.dataframe(pd.DataFrame(var_info))

    # Show coordinates
    if data.coords:
        container.write('**Coordinates:**')
        coord_info = []
        for coord_name, coord in data.coords.items():
            coord_info.append({'Coordinate': coord_name, 'Dimensions': ', '.join(coord.dims), 'Type': str(coord.dtype)})
        container.dataframe(pd.DataFrame(coord_info))

    # Show attributes
    if data.attrs:
        container.write('**Attributes:**')
        container.json(data.attrs)


@show_traceback()
def display_variable_stats(array: xr.DataArray, container: Optional[Any] = None) -> None:
    """
    Display basic statistics for a DataArray if it's numeric.

    Args:
        array: xarray.DataArray to compute stats for
        container: Streamlit container to render in (if None, uses st directly)
    """
    if container is None:
        container = st

    try:
        if np.issubdtype(array.dtype, np.number):
            stats_cols = container.columns(4)
            stats_cols[0].metric('Min', float(array.min().values))
            stats_cols[1].metric('Max', float(array.max().values))
            stats_cols[2].metric('Mean', float(array.mean().values))
            stats_cols[3].metric('Std', float(array.std().values))
    except Exception:
        pass


@show_traceback()
def aggregate_dimensions(
        array: xr.DataArray, agg_dims: List[str], agg_method: str, container: Optional[Any] = None
) -> xr.DataArray:
    """
    Aggregate a DataArray over specified dimensions using a specified method.

    Args:
        array: xarray.DataArray to aggregate
        agg_dims: List of dimension names to aggregate over
        agg_method: Aggregation method ('mean', 'sum', 'min', 'max', 'std', 'median')
        container: Streamlit container for displaying messages

    Returns:
        Aggregated DataArray
    """
    if container is None:
        container = st

    # Filter out any dimensions that don't exist in the array
    valid_agg_dims = [dim for dim in agg_dims if dim in array.dims]

    # If there are no valid dimensions to aggregate over, just return the original array
    if not valid_agg_dims:
        return array

    # Apply the selected aggregation method
    try:
        if agg_method == 'mean':
            result = array.mean(dim=valid_agg_dims)
        elif agg_method == 'sum':
            result = array.sum(dim=valid_agg_dims)
        elif agg_method == 'min':
            result = array.min(dim=valid_agg_dims)
        elif agg_method == 'max':
            result = array.max(dim=valid_agg_dims)
        elif agg_method == 'std':
            result = array.std(dim=valid_agg_dims)
        elif agg_method == 'median':
            result = array.median(dim=valid_agg_dims)
        elif agg_method == 'var':
            result = array.var(dim=valid_agg_dims)
        else:
            container.warning(f"Unknown aggregation method: {agg_method}. Using 'mean' instead.")
            result = array.mean(dim=valid_agg_dims)

        # If the aggregation removed all dimensions, ensure result has correct shape
        if len(result.dims) == 0:
            # Convert scalar result to 0D DataArray
            result = xr.DataArray(result.values, name=array.name, attrs=array.attrs)

        return result
    except Exception as e:
        container.error(f'Error during aggregation: {str(e)}')
        return array  # Return original array if aggregation fails


@show_traceback()
def create_dimension_selector(
        array: xr.DataArray,
        dim: str,
        container: Optional[Any] = None,
        unique_key: str = '',
) -> int:
    """
    Create a dimension selector (dropdown or slider) for a given dimension of an xarray.

    Args:
        array: The xarray DataArray to select from
        dim: The dimension name to create a selector for
        container: The Streamlit container to render in (if None, uses st)
        unique_key: A unique key suffix to prevent widget conflicts

    Returns:
        The selected index for the dimension
    """
    if container is None:
        container = st

    key_suffix = f'_{unique_key}' if unique_key else ''

    # Get dimension size
    dim_size = array.sizes[dim]

    # Default to middle value
    default_idx = dim_size // 2

    # Check if this dimension has coordinates
    if dim in array.coords:
        values = array.coords[dim].values

        # Use dropdown if fewer than 100 values, slider otherwise
        if len(values) < 100:
            # Use dropdown with actual coordinate values
            options = list(values)

            selected_value = container.selectbox(
                f'{dim}',
                options,
                index=default_idx,
                help=f'Select value for {dim}',
                key=f'dim_select_{dim}{key_suffix}',
            )

            # Find the index of the selected value
            if np.issubdtype(values.dtype, np.number):
                # For numeric values, find the closest index
                selected_idx = np.abs(values - selected_value).argmin()
            else:
                # For non-numeric values (strings, etc), find exact match
                try:
                    selected_idx = np.where(values == selected_value)[0][0]
                except Exception:
                    # Fallback if exact match fails
                    selected_idx = default_idx
        else:
            # Use slider for dimensions with many values
            selected_idx = container.slider(
                f'{dim}',
                0,
                dim_size - 1,
                default_idx,
                help=f'Position on {dim} dimension ({values[0]} to {values[-1]})',
                key=f'dim_slider_{dim}{key_suffix}',
                )

            # Show the selected value for context
            container.caption(f'Selected: {values[selected_idx]}')
    else:
        # No coordinates, use integer slider
        selected_idx = container.slider(
            f'{dim} index',
            0,
            dim_size - 1,
            default_idx,
            help=f'Position on {dim} dimension (by index)',
            key=f'dim_slider_idx_{dim}{key_suffix}',
            )

    return selected_idx


@show_traceback()
def create_dimension_selectors(
        array: xr.DataArray, slice_dims: List[str], container: Optional[Any] = None, unique_key: str = ''
) -> Dict[str, int]:
    """
    Create selectors for multiple dimensions and organize them in a grid layout.

    Args:
        array: The xarray DataArray to select from
        slice_dims: List of dimension names to create selectors for
        container: The Streamlit container to render in (if None, uses st)
        unique_key: A unique key suffix to prevent widget conflicts

    Returns:
        Dictionary mapping dimension names to selected indices
    """
    if container is None:
        container = st

    slice_indexes = {}

    if len(slice_dims) > 0:
        with container.expander('Dimension Values', expanded=True):
            # Calculate optimal number of columns based on number of dimensions
            num_cols = min(3, len(slice_dims))  # Max 3 columns to keep things readable

            # Create a grid layout of selectors
            for i in range(0, len(slice_dims), num_cols):
                # Create a new row of columns
                cols = container.columns(num_cols)

                # Fill the row with selectors
                for j in range(num_cols):
                    col_idx = i + j
                    if col_idx < len(slice_dims):
                        dim = slice_dims[col_idx]
                        with cols[j]:
                            slice_indexes[dim] = create_dimension_selector(array, dim, cols[j], f'{unique_key}_{i}_{j}')

    return slice_indexes


@show_traceback()
def plot_scalar(array: xr.DataArray, container: Optional[Any] = None) -> None:
    """
    Plot a scalar (0-dimensional) DataArray.

    Args:
        array: xarray.DataArray with 0 dimensions
        container: Streamlit container to render in (if None, uses st directly)
    """
    if container is None:
        container = st

    container.metric('Value', float(array.values))


@show_traceback()
def create_plot_style_picker(
        key_prefix: str, num_series: int = 1, container: Optional[Any] = None, series_labels: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Reusable plot style picker component for customizing plot colors and styles.
    Handles any number of data series with proper labeling.

    Args:
        key_prefix: Unique prefix for session state keys
        num_series: Number of data series to create colors for
        container: Streamlit container to render in
        series_labels: Optional list of labels for each series (e.g. dimension values)

    Returns:
        Dict with style settings
    """
    if container is None:
        container = st

    # Use provided labels or create generic ones
    if series_labels is None or len(series_labels) != num_series:
        series_labels = [f'Series {i + 1}' for i in range(num_series)]

    # Initialize in session state if needed
    style_key = f'{key_prefix}_plot_style'
    if style_key not in st.session_state:
        # Default Plotly colors for series
        default_colors = px.colors.qualitative.Plotly

        # Initialize with base settings
        st.session_state[style_key] = {
            'background_color': '#FFFFFF',
            'grid_color': '#EEEEEE',
            'title_color': '#000000',
            'colorscale': 'portland',  # For heatmaps
            'plot_height': 500,
            'series_colors': [],  # Start with empty list, will be populated below
        }

    styles = st.session_state[style_key]

    # Ensure we have enough colors for all series
    # This handles cases where num_series increases between renders
    if 'series_colors' not in styles:
        styles['series_colors'] = []

    # Get default colors to use
    default_colors = px.colors.qualitative.Plotly

    # Extend series_colors list if needed
    while len(styles['series_colors']) < num_series:
        idx = len(styles['series_colors']) % len(default_colors)
        styles['series_colors'].append(default_colors[idx])

    # Create expandable section for style settings
    with container.expander('Customize Plot Style', expanded=False):
        # Create tabs for different style categories
        color_tab, layout_tab = st.tabs(['Colors', 'Layout'])

        with color_tab:
            # Create color presets
            presets = {
                'Default': {
                    'background_color': '#FFFFFF',
                    'grid_color': '#EEEEEE',
                    'title_color': '#000000',
                    'colorscale': 'portland',
                    'series_colors': px.colors.qualitative.Plotly,
                },
                'Dark Mode': {
                    'background_color': '#1E1E1E',
                    'grid_color': '#333333',
                    'title_color': '#FFFFFF',
                    'colorscale': 'Viridis',
                    'series_colors': px.colors.qualitative.Light24,
                },
                'Pastel': {
                    'background_color': '#F9F9F9',
                    'grid_color': '#EEEEEE',
                    'title_color': '#333333',
                    'colorscale': 'Pastel',
                    'series_colors': px.colors.qualitative.Pastel,
                },
            }

            col1, col2 = st.columns([2, 1])

            # Preset selector
            selected_preset = col1.selectbox('Color Preset', list(presets.keys()), key=f'{key_prefix}_preset')

            # Apply preset button
            if col2.button('Apply', key=f'{key_prefix}_apply_preset'):
                preset = presets[selected_preset]
                styles['background_color'] = preset['background_color']
                styles['grid_color'] = preset['grid_color']
                styles['title_color'] = preset['title_color']
                styles['colorscale'] = preset['colorscale']

                # Update series colors with the preset colors
                preset_colors = preset['series_colors']
                for i in range(num_series):
                    styles['series_colors'][i] = preset_colors[i % len(preset_colors)]

            st.markdown('##### Background & Layout')
            col1, col2 = st.columns(2)

            # Basic color pickers
            styles['background_color'] = col1.color_picker(
                'Background', styles['background_color'], key=f'{key_prefix}_bg_color'
            )

            styles['grid_color'] = col2.color_picker('Grid Lines', styles['grid_color'], key=f'{key_prefix}_grid_color')

            styles['title_color'] = col1.color_picker('Title', styles['title_color'], key=f'{key_prefix}_title_color')

            # Colorscale for heatmaps
            colorscales = [
                'portland',
                'Viridis',
                'Plasma',
                'Inferno',
                'Blues',
                'Reds',
                'Greens',
                'YlOrRd',
                'Cividis',
                'RdBu',
            ]
            styles['colorscale'] = col2.selectbox(
                'Heatmap Scale',
                colorscales,
                index=colorscales.index(styles['colorscale']) if styles['colorscale'] in colorscales else 0,
                key=f'{key_prefix}_colorscale',
            )

            # Series colors - dynamically create UI based on num_series
            st.markdown('##### Data Series Colors')

            # Determine how many to show per row (3 is a good default)
            series_per_row = 3

            # Create rows of colors with up to series_per_row columns
            for i in range(0, num_series, series_per_row):
                # Create columns for this row
                cols = st.columns(series_per_row)

                # Fill each column with a color picker if we have a series for it
                for j in range(series_per_row):
                    series_idx = i + j

                    # Only create a color picker if we have a series at this index
                    if series_idx < num_series:
                        # Use the provided label instead of generic "Series N"
                        label = series_labels[series_idx]
                        styles['series_colors'][series_idx] = cols[j].color_picker(
                            label,
                            styles['series_colors'][series_idx],
                            key=f'{key_prefix}_series_{series_idx}_color',
                        )

            # Option to reset series colors to sequential defaults
            if st.button('Reset Series Colors', key=f'{key_prefix}_reset_series'):
                default_colors = px.colors.qualitative.Plotly
                for i in range(num_series):
                    styles['series_colors'][i] = default_colors[i % len(default_colors)]

        with layout_tab:
            # Additional layout options
            st.markdown('##### Plot Size')
            styles['plot_height'] = st.slider(
                'Height (px)',
                min_value=300,
                max_value=1000,
                value=styles.get('plot_height', 500),
                key=f'{key_prefix}_plot_height',
            )

    # Always ensure we return the correct number of colors
    # In case the number of series decreased, truncate the list
    styles['series_colors'] = styles['series_colors'][:num_series]

    return styles


@show_traceback()
def plot_1d(array: xr.DataArray, var_name: str, container: Optional[Any] = None) -> None:
    """Plot with color customization"""
    if container is None:
        container = st

    dim = list(array.dims)[0]


    # Plot type selector
    plot_type = container.selectbox('Plot type:', ['Line', 'Bar', 'Histogram', 'Area'], key=f'plot_type_1d_{var_name}')

    customize_colors = container.checkbox('Customize colors', key=f'customize_colors_1d_{var_name}')
    if customize_colors:
        # Color configuration - get just one series since this is a 1D plot
        colors = create_plot_style_picker(f'plot_1d_{var_name}', num_series=1, series_labels=[var_name])

    # Create figure based on selected plot type
    if plot_type == 'Line':
        fig = px.line(
            x=array[dim].values, y=array.values, labels={'x': dim, 'y': var_name}, title=f'{var_name} by {dim}'
        )

        if customize_colors:
            fig.update_traces(line_color=colors['series_colors'][0])

    elif plot_type == 'Bar':
        # Code as before, applying colors
        df = pd.DataFrame({dim: array[dim].values, 'value': array.values})
        fig = px.bar(df, x=dim, y='value', labels={'value': var_name}, title=f'{var_name} by {dim}')

        if customize_colors:
            fig.update_traces(marker_color=colors['series_colors'][0])

    elif plot_type == 'Histogram':
        fig = px.histogram(array.values, title=f'Histogram of {var_name}')

        if customize_colors:
            fig.update_traces(marker_color=colors['series_colors'][0])

    elif plot_type == 'Area':
        fig = px.area(
            x=array[dim].values, y=array.values, labels={'x': dim, 'y': var_name}, title=f'{var_name} by {dim}'
        )

        if customize_colors:
            fig.update_traces(line_color=colors['series_colors'][0], fillcolor=colors['series_colors'][0])

    if customize_colors:
        # Apply common styling to any plot type
        fig.update_layout(
            plot_bgcolor=colors['background_color'],
            paper_bgcolor=colors['background_color'],
            xaxis=dict(gridcolor=colors['grid_color']),
            yaxis=dict(gridcolor=colors['grid_color']),
            title_font_color=colors['title_color'],
        )

    # Show the plot and remaining functionality
    container.plotly_chart(fig, use_container_width=True)

    # For 1D data, we can also offer some basic statistics
    if container.checkbox('Show statistics', key=f'show_stats_{var_name}'):
        try:
            stats = pd.DataFrame(
                {
                    'Statistic': ['Min', 'Max', 'Mean', 'Median', 'Std', 'Sum'],
                    'Value': [
                        float(array.min().values),
                        float(array.max().values),
                        float(array.mean().values),
                        float(np.median(array.values)),
                        float(array.std().values),
                        float(array.sum().values),
                    ],
                }
            )
            container.dataframe(stats, use_container_width=True)
        except Exception as e:
            container.warning(f'Could not compute statistics: {str(e)}')


@show_traceback()
def plot_nd(array: xr.DataArray, var_name: str, container: Optional[Any] = None) -> Tuple[xr.DataArray, Optional[Dict]]:
    """
    Plot a multi-dimensional DataArray with interactive dimension selectors.
    Supports multiple plot types and dimension aggregation.

    Args:
        array: xarray.DataArray with 2+ dimensions
        var_name: Name of the variable being plotted
        container: Streamlit container to render in (if None, uses st directly)

    Returns:
        Tuple of (sliced array, selection dictionary)
    """
    if container is None:
        container = st

    dims = list(array.dims)

    # Use tabs for main sections
    dim_tab, viz_tab = container.tabs(['Dimension Settings', 'Visualization Settings'])

    # === DIMENSION SETTINGS TAB ===
    with dim_tab:
        # Use columns for dimension handling
        agg_col1, agg_col2 = dim_tab.columns([3, 2])

        with agg_col1:
            # Multi-select for dimensions to aggregate
            agg_dims = st.multiselect(
                'Dimensions to aggregate:',
                dims,
                default=[],
                help='Select dimensions to aggregate',
            )

        with agg_col2:
            # Aggregation method selection
            agg_method = st.selectbox(
                'Method:',
                ['mean', 'sum', 'min', 'max', 'std', 'median', 'var'],
                index=0,
            )

    # Apply aggregation if dimensions were selected
    if agg_dims:
        orig_dims = dims.copy()
        array = aggregate_dimensions(array, agg_dims, agg_method, container)
        dims = list(array.dims)

        # Show information about the aggregation
        removed_dims = [dim for dim in orig_dims if dim not in dims]
        if removed_dims:
            msg = f'Applied {agg_method} over: {", ".join(removed_dims)}'
            container.info(msg)

    # If no dimensions left after aggregation, show scalar result
    if len(dims) == 0:
        plot_scalar(array, container)
        return array, None

    # If one dimension left after aggregation, use 1D plotting
    if len(dims) == 1:
        plot_1d(array, var_name, container)
        return array, None

    # === VISUALIZATION SETTINGS TAB ===
    with viz_tab:
        # Use columns for visualization settings
        viz_col1, viz_col2 = viz_tab.columns(2)

        with viz_col1:
            # Choose which dimension to put on x-axis
            x_dim = st.selectbox('X dimension:', dims, index=0)

            # Choose which dimension to put on y-axis if we have at least 2 dimensions
            remaining_dims = [d for d in dims if d != x_dim]
            y_dim = None
            if len(remaining_dims) > 0:
                y_dim_options = ['None'] + remaining_dims
                y_dim_selection = st.selectbox('Y dimension:', y_dim_options, index=1)
                if y_dim_selection != 'None':
                    y_dim = y_dim_selection

        with viz_col2:
            # Add plot type selector
            plot_types = ['Heatmap', 'Line', 'Stacked Bar', 'Grouped Bar']
            if y_dim is None:
                # Remove heatmap option if there's no Y dimension
                plot_types = [pt for pt in plot_types if pt != 'Heatmap']
                default_idx = 0  # Default to Line for 1D
            else:
                default_idx = 0  # Default to Heatmap for 2D

            plot_type = st.selectbox('Plot type:', plot_types, index=default_idx)

    # If we have more than the selected dimensions, let user select values for other dimensions
    # Calculate which dimensions need slicers
    slice_dims = [d for d in dims if d not in ([x_dim] if y_dim is None else [x_dim, y_dim])]
    slice_indexes = {}

    # Create a more compact layout for dimension dropdown selectors
    slice_indexes = create_dimension_selectors(array, slice_dims, container, 'key')

    # Create slice dictionary for selection
    slice_dict = {dim: slice_indexes[dim] for dim in slice_dims}

    # Select the data to plot
    if slice_dims:
        array_slice = array.isel(slice_dict)
    else:
        array_slice = array

    # Visualization depends on the selected plot type and dimensions
    container.subheader('Plot')

    # ===== ADD STYLE PICKER INTEGRATION HERE =====
    # Calculate number of series for the style picker
    num_series = 1
    series_labels = None

    if y_dim is not None:
        # Get the actual values from the y dimension to use as labels
        y_values = array_slice[y_dim].values
        num_series = len(y_values)

        # Format the values as readable labels
        # Convert values to strings and add the dimension name for context
        series_labels = [str(val) for val in y_values]

    # Create unique key for this plot
    style_key = f'plot_nd_{var_name}_{"_".join(dims)}'

    # Create the style picker
    customize_colors = container.checkbox('Customize colors', key=f'customize_colors_1d_{var_name}')
    if customize_colors:
        # Color configuration - get just one series since this is a 1D plot
        style_settings = create_plot_style_picker(style_key, num_series=num_series, series_labels=series_labels)

    if y_dim is not None:
        # 2D visualization
        if plot_type == 'Heatmap':
            # Heatmap visualization
            fig = px.imshow(
                array_slice.transpose(y_dim, x_dim).values,
                x=array_slice[x_dim].values,
                y=array_slice[y_dim].values,
                color_continuous_scale=style_settings['colorscale'] if customize_colors else None,  # Use custom colorscale
                labels={'x': x_dim, 'y': y_dim, 'color': var_name},
                title=f'{var_name} by {x_dim} and {y_dim}',
            )

        elif plot_type == 'Line':
            # Line plot with multiple lines (one per y-dimension value)
            fig = go.Figure()

            # Convert to dataframe for easier plotting
            df = array_slice.to_dataframe(name='value').reset_index()

            # Group by y-dimension for multiple lines
            for i, y_val in enumerate(array_slice[y_dim].values):
                df_subset = df[df[y_dim] == y_val]
                fig.add_trace(
                    go.Scatter(
                        x=df_subset[x_dim],
                        y=df_subset['value'],
                        mode='lines',
                        name=str(y_val),
                        line=dict(color=style_settings['series_colors'][i % len(style_settings['series_colors'])]) if customize_colors else None,

                    )
                )
            fig.update_layout(
                title=f'{var_name} by {x_dim} and {y_dim}',
                xaxis_title=x_dim,
                yaxis_title=var_name,
                legend_title=y_dim,
            )

        elif plot_type == 'Stacked Bar':
            # Stacked bar chart
            # Convert to dataframe for easier plotting
            df = array_slice.to_dataframe(name='value').reset_index()
            df = df.fillna(0)  # Fixes issues with stacking

            fig = px.bar(
                df,
                x=x_dim,
                y='value',
                color=y_dim,
                barmode='relative',
                title=f'{var_name} by {x_dim} and {y_dim}',
                labels={'value': var_name, x_dim: x_dim, y_dim: y_dim},
                color_discrete_sequence=style_settings['series_colors'] if customize_colors else None,
            )

        elif plot_type == 'Grouped Bar':
            # Grouped bar chart
            # Convert to dataframe for easier plotting
            df = array_slice.to_dataframe(name='value').reset_index()

            fig = px.bar(
                df,
                x=x_dim,
                y='value',
                color=y_dim,
                barmode='group',
                title=f'{var_name} by {x_dim} and {y_dim}',
                labels={'value': var_name, x_dim: x_dim, y_dim: y_dim},
                color_discrete_sequence=style_settings['series_colors'] if customize_colors else None,
            )

    else:
        # 1D visualization after slicing (no y_dim)
        if plot_type == 'Line':
            fig = px.line(
                x=array_slice[x_dim].values,
                y=array_slice.values,
                labels={'x': x_dim, 'y': var_name},
                title=f'{var_name} by {x_dim}',
                color=style_settings['series_colors'][0] if customize_colors else None,
            )

        elif plot_type in ['Stacked Bar', 'Grouped Bar']:  # Both are the same for 1D
            # Create a dataframe for the bar chart
            df = pd.DataFrame({x_dim: array_slice[x_dim].values, 'value': array_slice.values})

            fig = px.bar(
                df,
                x=x_dim,
                y='value',
                labels={'x': x_dim, 'y': var_name},
                title=f'{var_name} by {x_dim}',
                color=style_settings['series_colors'][0] if customize_colors else None
            )

    if customize_colors:
        # Apply common layout settings
        fig.update_layout(
            height=style_settings['plot_height'],
            paper_bgcolor=style_settings['background_color'],
            plot_bgcolor=style_settings['background_color'],
            font_color=style_settings['title_color'],
            xaxis=dict(gridcolor=style_settings['grid_color']),
            yaxis=dict(gridcolor=style_settings['grid_color']),
        )

    container.plotly_chart(fig, use_container_width=True)
    return array_slice, slice_dict


@show_traceback()
def display_data_preview(array: xr.DataArray, container: Optional[Any] = None) -> pd.DataFrame:
    """
    Display a preview of the data as a dataframe.

    Args:
        array: xarray.DataArray to preview
        container: Streamlit container to render in (if None, uses st directly)

    Returns:
        DataFrame containing the preview data
    """
    if container is None:
        container = st

    try:
        # Limit to first 1000 elements for performance
        preview_data = array
        total_size = np.prod(preview_data.shape)

        if total_size > 1000:
            container.warning(f'Data is large ({total_size} elements). Showing first 1000 elements.')
            # Create a slice dict to get first elements from each dimension
            preview_slice = {}
            remaining = 1000
            for dim in preview_data.dims:
                dim_size = preview_data.sizes[dim]
                take = min(dim_size, max(1, int(remaining ** (1 / len(preview_data.dims)))))
                preview_slice[dim] = slice(0, take)
                remaining = remaining // take

            preview_data = preview_data.isel(preview_slice)

        # Convert to dataframe and display
        df = preview_data.to_dataframe()
        container.dataframe(df)
        return df
    except Exception as e:
        container.error(f'Could not convert to dataframe: {str(e)}')
        return pd.DataFrame()


@show_traceback()
def filter_xarray_by_coords(
        data: Union[xr.Dataset, xr.DataArray], container: Optional[Any] = None, key_prefix: str = '', expanded: bool = False
) -> Tuple[Union[xr.Dataset, xr.DataArray], bool, Dict]:
    """
    Create interactive coordinate filters for xarray data and apply them.

    Args:
        data: xarray Dataset or DataArray to filter
        container: Streamlit container (defaults to st)
        key_prefix: Prefix for Streamlit widget keys to avoid conflicts
        expanded: Whether the expander starts expanded

    Returns:
        Tuple of (filtered_data, filters_applied, filter_specs)
    """
    if container is None:
        container = st

    # Keep track of original data to return if no filters are applied
    filtered_data = data
    filters_applied = False

    # Create filter UI in an expander
    with container.expander('Filter Data', expanded=expanded):
        # Coordinate filtering
        container.markdown('### Filter by Coordinates')

        # Identify all coordinates in the data
        coords = list(data.coords)

        # Group coordinates by type
        dim_coords = [c for c in coords if c in data.dims]
        non_dim_coords = [c for c in coords if c not in data.dims]

        # Let user select which coordinates to filter by
        selected_coords = container.multiselect(
            'Select coordinates to filter by:', coords, default=[], key=f'{key_prefix}_selected_coords'
        )

        # For each selected coordinate, provide appropriate filter options
        coord_filters = {}
        for coord in selected_coords:
            container.markdown(f'**Filter by {coord}:**')

            # Get unique values for this coordinate
            coord_values = data.coords[coord].values

            # Handle different types of coordinates differently
            if np.issubdtype(coord_values.dtype, np.datetime64):
                # For datetime coordinates, use date range sliders
                min_date = pd.to_datetime(coord_values.min())
                max_date = pd.to_datetime(coord_values.max())

                date_col1, date_col2 = container.columns(2)
                start_date = date_col1.date_input(
                    f'Start date for {coord}',
                    value=min_date,
                    min_value=min_date,
                    max_value=max_date,
                    key=f'{key_prefix}_{coord}_start',
                )
                end_date = date_col2.date_input(
                    f'End date for {coord}',
                    value=max_date,
                    min_value=min_date,
                    max_value=max_date,
                    key=f'{key_prefix}_{coord}_end',
                )

                # Convert to numpy datetime64
                start_np = np.datetime64(pd.Timestamp(start_date))
                end_np = np.datetime64(pd.Timestamp(end_date))

                coord_filters[coord] = slice(start_np, end_np)

            elif np.issubdtype(coord_values.dtype, np.number):
                # For numeric coordinates, use number range sliders
                try:
                    min_val = float(coord_values.min())
                    max_val = float(coord_values.max())
                except:
                    container.error(f'Error getting min/max for {coord}. Using defaults.')
                    min_val, max_val = 0.0, 1.0

                num_col1, num_col2 = container.columns(2)
                start_val = num_col1.number_input(
                    f'Min value for {coord}',
                    value=min_val,
                    min_value=min_val,
                    max_value=max_val,
                    key=f'{key_prefix}_{coord}_min',
                )
                end_val = num_col2.number_input(
                    f'Max value for {coord}',
                    value=max_val,
                    min_value=min_val,
                    max_value=max_val,
                    key=f'{key_prefix}_{coord}_max',
                )

                coord_filters[coord] = slice(start_val, end_val)

            else:
                # For categorical/string/other coordinates, use multiselect
                unique_values = np.unique(coord_values)
                selected_values = container.multiselect(
                    f'Select values for {coord}',
                    unique_values,
                    default=list(unique_values),
                    key=f'{key_prefix}_{coord}_values',
                )

                coord_filters[coord] = selected_values

        # Add a button to apply filters
        apply_filters = container.button('Apply Filters', key=f'{key_prefix}_apply_filters')

    # Apply filters if requested
    if apply_filters and coord_filters:
        try:
            # Separate dimension and non-dimension filters
            dim_filters = {dim: filter_value for dim, filter_value in coord_filters.items() if dim in dim_coords}
            non_dim_filters = {
                coord: filter_value for coord, filter_value in coord_filters.items() if coord in non_dim_coords
            }

            # Apply dimension filters using sel()
            if dim_filters:
                filtered_data = filtered_data.sel(**dim_filters)

            # Apply non-dimension coordinate filters using where()
            for coord, filter_value in non_dim_filters.items():
                if isinstance(filter_value, list):
                    # Handle list of selected values
                    if filter_value:  # Only if not empty
                        filtered_data = filtered_data.where(filtered_data.coords[coord].isin(filter_value), drop=True)
                else:
                    # Handle slice (range) selection
                    start, end = filter_value.start, filter_value.stop
                    filtered_data = filtered_data.where(
                        (filtered_data.coords[coord] >= start) & (filtered_data.coords[coord] <= end), drop=True
                    )

            filters_applied = True

        except Exception as e:
            container.error(f'Error applying filters: {str(e)}')
            # Keep original data if filtering fails
            filtered_data = data

    # Return filtered data, whether filters were applied, and filter specifications
    return filtered_data, filters_applied, coord_filters


@show_traceback()
def xarray_explorer(
        data: Union[xr.Dataset, xr.DataArray],
        custom_plotters: Optional[Dict[str, Callable]] = None,
        container: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    A modular xarray explorer for both DataArrays and Datasets with coordinate filtering.

    Args:
        data: xarray.Dataset or xarray.DataArray
        custom_plotters: Dictionary of custom plotting functions by dimension
        container: Streamlit container to render in (if None, uses st directly)

    Returns:
        Dictionary containing information about the current state
    """
    if container is None:
        container = st

    # Convert scenario dimension to string to ensure categorical plots
    if 'scenario' in data.dims:
        data = data.assign_coords({'scenario': data.coords['scenario'].astype(str)})

    filtered_data, filters_applied, filter_specs = filter_xarray_by_coords(data, container, key_prefix='explorer')

    # Determine if we're working with Dataset or DataArray
    is_dataset = isinstance(data, xr.Dataset)

    # Variable selection for Dataset or direct visualization for DataArray
    if is_dataset:
        # Variable selection
        selected_var = container.selectbox('Select variable:', list(filtered_data.data_vars))
        array_to_plot = filtered_data[selected_var]
    else:
        # If DataArray, use directly
        array_to_plot = filtered_data
        selected_var = filtered_data.name if filtered_data.name else 'Data'

    # Initialize result dictionary
    result = {
        'data': data,  # Original data
        'filtered_data': filtered_data,  # Data after filters applied
        'selected_array': array_to_plot,
        'selected_var': selected_var,
        'sliced_array': None,
        'slice_dict': None,
    }

    # Visualization section
    container.subheader('Visualization')

    # Determine available visualization options based on dimensions
    dims = list(array_to_plot.dims)
    ndim = len(dims)

    # Get the appropriate plotter function
    plotters = {'scalar': plot_scalar, '1d': plot_1d, 'nd': plot_nd}

    # Override with custom plotters if provided
    if custom_plotters:
        plotters.update(custom_plotters)

    # Different visualization options based on dimensionality
    if ndim == 0:
        # Scalar value
        plotters['scalar'](array_to_plot, container)
    elif ndim == 1:
        # 1D data
        plotters['1d'](array_to_plot, selected_var, container)
    else:
        # 2D+ data
        sliced_array, slice_dict = plotters['nd'](array_to_plot, selected_var, container)
        result['sliced_array'] = sliced_array
        result['slice_dict'] = slice_dict

    # Data preview section
    with container.expander('Data Preview', expanded=False):
        display_data_preview(array_to_plot, container)

    # Download options
    download_format = container.selectbox('Download format', ['CSV', 'NetCDF', 'Excel'])

    if container.button('Download filtered data'):
        download_data(
            array_to_plot if result['sliced_array'] is None else result['sliced_array'],
            selected_var,
            download_format,
            container,
        )

    container.subheader('Data Information')
    display_data_info(filtered_data, container)  # Show info for filtered data

    # Display variable information
    container.subheader(f'Variable: {selected_var}')
    display_variable_stats(array_to_plot, container)

    return result


def create_example_data():
    """
    Create example xarray data for demonstration.

    Returns:
        xr.Dataset: Example dataset with multiple variables
    """
    # Create a simple dataset with temperature, precipitation, and pressure
    # across time, latitude, and longitude

    # Create dimensions
    time = pd.date_range('2020-01-01', periods=30, freq='D')
    lat = np.linspace(25, 35, 10)
    lon = np.linspace(-90, -80, 10)

    # Create data variables
    np.random.seed(42)  # For reproducibility

    # Temperature with daily cycle and spatial patterns
    temp_data = 15 + 10 * np.sin(np.linspace(0, 2*np.pi, len(time)))[:, np.newaxis, np.newaxis] + \
                np.random.normal(0, 1, size=(len(time), len(lat), len(lon)))

    # Precipitation with some random storm patterns
    precip_data = np.random.exponential(2, size=(len(time), len(lat), len(lon)))

    # Pressure with a gradient across latitude
    pressure_base = 1013.25  # Standard sea-level pressure in hPa
    pressure_data = np.ones((len(time), len(lat), len(lon))) * pressure_base
    # Add latitude gradient
    for i, lat_val in enumerate(lat):
        pressure_data[:, i, :] += (lat_val - 30) * 0.5
    # Add time variation
    for t in range(len(time)):
        pressure_data[t, :, :] += np.sin(t/5) * 5

    # Random noise
    pressure_data += np.random.normal(0, 1, size=(len(time), len(lat), len(lon)))

    # Create dataset
    ds = xr.Dataset(
        data_vars=dict(
            temperature=(['time', 'lat', 'lon'], temp_data,
                         {'units': 'celsius', 'long_name': 'Air temperature'}),
            precipitation=(['time', 'lat', 'lon'], precip_data,
                           {'units': 'mm/day', 'long_name': 'Daily precipitation'}),
            pressure=(['time', 'lat', 'lon'], pressure_data,
                      {'units': 'hPa', 'long_name': 'Atmospheric pressure'})
        ),
        coords=dict(
            lon=(['lon'], lon, {'units': 'degrees_east', 'long_name': 'Longitude'}),
            lat=(['lat'], lat, {'units': 'degrees_north', 'long_name': 'Latitude'}),
            time=(['time'], time)
        ),
        attrs=dict(
            description='Example weather dataset',
            source='Synthetic data',
            created=pd.Timestamp.now().isoformat()
        )
    )

    # Add some derived data
    ds['heat_index'] = ds['temperature'] + 0.05 * ds['pressure'] / 10
    ds.heat_index.attrs = {'units': 'celsius', 'long_name': 'Heat index'}

    return ds


def load_dataset_from_file(file):
    """
    Load xarray dataset from an uploaded file.

    Args:
        file: Uploaded file object

    Returns:
        xr.Dataset or xr.DataArray
    """
    try:
        # Save uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nc') as tmp:
            tmp.write(file.getvalue())
            tmp_path = tmp.name

        # Open with xarray
        data = xr.open_dataset(tmp_path)

        # Clean up temp file
        os.unlink(tmp_path)

        return data
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None


def main():
    """Main function for the xarray explorer app"""
    # Set page config
    st.set_page_config(
        page_title="Xarray Explorer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Title and introduction
    st.title("📊 Xarray Dataset Explorer")

    st.markdown("""
    This app helps you explore and visualize xarray datasets. You can:

    - Upload your own NetCDF file or use the example data
    - Filter data by coordinates
    - Create visualizations based on dimensions
    - Download filtered data in various formats
    - View detailed statistics and information
    """)

    # Create sidebar for data source selection
    st.sidebar.title("Data Source")

    data_source = st.sidebar.radio(
        "Select data source:",
        ["Upload NetCDF file", "Use example data"]
    )

    # Initialize data variable
    data = None

    # Handle data source selection
    if data_source == "Upload NetCDF file":
        uploaded_file = st.sidebar.file_uploader("Upload a NetCDF file", type=["nc", "netcdf", 'nc4', 'zarr'])

        if uploaded_file is not None:
            data = load_dataset_from_file(uploaded_file)
            if data is not None:
                st.sidebar.success(f"Successfully loaded dataset with {len(data.data_vars)} variables")
        else:
            st.info("Please upload a NetCDF file to begin exploration")
    else:
        data = create_example_data()
        st.sidebar.info("Using example weather dataset (temperature, precipitation, pressure)")

    # Add options to customize plots
    if data is not None:
        # Main content
        st.markdown("## Dataset Explorer")

        # Use the xarray explorer if data is loaded
        xarray_explorer(data)


if __name__ == "__main__":
    main()