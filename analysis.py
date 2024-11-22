import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as patches
import numpy as np
from collections import defaultdict


def plot_correlation_heatmap(X, include_fixed=True):
    """
    Plots correlation heatmaps for features at the earliest and latest timesteps.

    :param X: DataFrame containing the features to plot.
    :param include_fixed: Whether to include fixed features in the correlation.
    """

    # Define patterns for the earliest and latest timesteps
    earliest_pattern = "_Tmin_-300"
    latest_pattern = "_Tmin_0"
    print(f'Earliest Pattern: {earliest_pattern}')
    print(f'Latest Pattern: {latest_pattern}')
    
    # Filter columns
    X = X[[col for col in X.columns if 'regulations' not in col and 'eobt' not in col and 'offblock' not in col and 'fltstate' not in col and 'modeltyp' not in col]]
    earliest_features = X.filter(regex=f"{earliest_pattern}")
    latest_features = X.filter(regex=f"{latest_pattern}")
    
    # Select fixed features and drop specific columns
    fixed_features = X.loc[:, ~X.columns.str.contains('_Tmin')]
    fixed_features = fixed_features.drop(['atfmdelay'], axis=1)

    # Convert datetime columns to datetime format and exclude them from correlation
    datetime_cols = ['ETOT', 'EOBT', 'ETA', 'cbasentry', 'TSAT', 'TOBT']
    for col in datetime_cols:
        fixed_features[col] = pd.to_datetime(fixed_features[col], errors='coerce')
    # print(f'{fixed_features=}')
    
    # Concatenate fixed features with earliest and latest time-varying features
    earliest_features = pd.concat([fixed_features, earliest_features.reset_index(drop=True)], axis=1)
    latest_features = pd.concat([fixed_features, latest_features.reset_index(drop=True)], axis=1)
    
    # Select only numerical columns and drop rows with missing values for correlation computation
    earliest_features = earliest_features.select_dtypes(include=['number']).dropna()
    latest_features = latest_features.select_dtypes(include=['number']).dropna()
    earliest_features.columns = [col.replace(earliest_pattern, '').replace('_', ' ') for col in earliest_features.columns]
    latest_features.columns = [col.replace(latest_pattern, '').replace('_', ' ') for col in latest_features.columns]
    # Compute correlation matrices
    corr_earliest = abs(earliest_features.corr())
    corr_latest = abs(latest_features.corr())

    # Use a lightened green colormap
    light_seagreen_palette = sns.light_palette("seagreen", n_colors=5, reverse=False, as_cmap=True)

    # Create a figure with attractive settings
    fig, axs = plt.subplots(1, 2, figsize=(28, 12))
    plt.suptitle("Correlation Heatmaps for Earliest and Latest Time Steps", fontsize=20, weight='bold', color='#333')

    # Set heatmap color palette and annotations
    heatmap_kwargs = {
        "cmap": light_seagreen_palette, 
        "annot": True, 
        "fmt": ".2f", 
        "annot_kws": {"size": 15}, 
        "cbar_kws": {"shrink": 0.8, "aspect": 15}
    }

    # Plot the heatmaps with titles and adjusted tick size
    sns.heatmap(corr_earliest, ax=axs[0], **heatmap_kwargs)
    axs[0].set_title("Earliest Time Step (T-300)", fontsize=16, weight='bold')
    axs[0].tick_params(axis='x', rotation=90, labelsize=15 )
    axs[0].tick_params(axis='y', rotation=0, labelsize=15 )

    sns.heatmap(corr_latest, ax=axs[1], **heatmap_kwargs)
    axs[1].set_title("Latest Time Step (T-0)", fontsize=16, weight='bold')
    axs[1].tick_params(axis='x', rotation=90, labelsize=15 )
    axs[1].tick_params(axis='y', rotation=0, labelsize=15 )

    # Draw a rectangle around the "delay" row in each heatmap if "delay" is present
    for ax, corr_matrix in zip(axs, [corr_earliest, corr_latest]):
        if 'delay' in corr_matrix.index:
            delay_row_idx = corr_matrix.index.get_loc('delay')
            # Rectangle parameters: position (x, y), width, and height
            rect = patches.Rectangle(
                (0, delay_row_idx),  # (x, y) position
                len(corr_matrix.columns),  # Width (number of columns)
                1,  # Height of one row
                linewidth=2,
                edgecolor='red',
                facecolor='none'
            )
            ax.add_patch(rect)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit the main title
    plt.show()


def get_ETO_DEP(extended_df, plot=True):
    etolist = []
    TSATlist = []
    TOBTlist = []
    horizons =  range(-300, 5, 5)
    cbaserrors = {}
    for t in horizons:
        etolist.append(np.mean(np.abs(extended_df['delay']-extended_df[f'etodepdelay_Tmin_{t}'])))
        TSATlist.append(np.mean(np.abs(extended_df['delay']-extended_df[f'TSATdelay_Tmin_{t}'])))
        TOBTlist.append(np.mean(np.abs(extended_df['delay']+extended_df[f'TOBTdelay_Tmin_{t}'])))
        cbas = extended_df[f'timetoCBAS_Tmin_{t}'][0].total_seconds() / 60
        if cbas not in cbaserrors.keys():
             cbaserrors[cbas] = []
        cbaserrors[cbas].append( np.mean(np.abs(extended_df['delay']-extended_df[f'etodepdelay_Tmin_{t}'])))
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(horizons, etolist, label='ETO Delay', marker='o', linestyle='-')
        # plt.plot(horizons, TSATlist, label='TSAT Delay', marker='o', linestyle='-')
        # plt.plot(horizons, TOBTlist, label='TOBT Delay', marker='o', linestyle='-')


        plt.title("Mean Prediction Error vs Time to ATOT (bucketed in 5-minute intervals)")
        plt.xlabel("Time to ATOT")
        plt.ylabel("Mean Prediction Error (minutes)")
        plt.grid(True)


        plt.legend()
        plt.show()

    bucketed_errors = defaultdict(list)
    for time_in_minutes, errors in cbaserrors.items():
        bucket = int(time_in_minutes // 5) * 5   # Group into 5-minute buckets
        bucketed_errors[bucket].extend(errors)   # Aggregate errors within the bucket
    mean_errors_by_bucket = {bucket: np.nanmean(errors) for bucket, errors in bucketed_errors.items()}
    sorted_buckets = sorted(mean_errors_by_bucket.items())
    print(f'{sorted_buckets=}')
    times = [item[0] for item in sorted_buckets]
    errors = [item[1] for item in sorted_buckets]
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(times, errors, marker='o', linestyle='-')
        plt.title("Mean Absolute Prediction Error vs Time to CBAS Entry (bucketed in 5-minute intervals)")
        plt.xlabel("Time to CBAS Entry (minutes)")
        plt.ylabel("Mean Absolute Prediction Error (minutes)")
        ax = plt.gca()
        ax.invert_xaxis()

        plt.grid(True)
        plt.show()
    return etolist, {i[0]:i[1] for i in sorted_buckets}