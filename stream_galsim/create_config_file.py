import galstreams
import pandas as pd
import yaml
import os

## Import Milky Way streams from galstreams catalog and make summary
mws = galstreams.MWStreams(verbose=False, implement_Off=False)

# User specifies the StreamName (filter based on the Name column)
column_name = 'Name'
stream_name = "Pal5"
column_value = stream_name

# Convert to DataFrame
df = mws.summary

# Filter the DataFrame for the specific stream
filtered_df = df[df[column_name] == stream_name]

# Custom Data
user_coord = pd.DataFrame({
    'parameter': ['cluster','ra', 'dec', 'distance', 'pm_ra_cosdec', 'pm_dec', 'radial_velocity', 'frame'],
    'value': [f'{stream_name}',229, -0.124, 22.9, -2.296, -2.257, -58.7, 'icrs'],
    'unit': [None, 'deg', 'deg', 'kpc', 'mas/yr', 'mas/yr', 'km/s', None]
})

# Check if there is data after filtering
if filtered_df.empty:
    print(f"No data found for {stream_name}.")
else:
    # Convert filtered DataFrame to a list of dictionaries
    yaml_data = filtered_df.to_dict(orient="records")

    # Transform DataFrame into dictionary format with unit as suffix
    user_dict = {}
    for _, row in user_coord.iterrows():
        param_name = row['parameter']
        user_dict[param_name] = row['value']
        if pd.notna(row['unit']):  # Only add unit field if it's not NaN
            user_dict[f"{param_name}_unit"] = row['unit']

    # Append to YAML data
    # yaml_data.append({'user_coord'})
    yaml_data.append(user_dict)

    # Create the YAML filename based on the stream value
    config_dir = "config"
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    yaml_filename = os.path.join(config_dir, f"galstreams_{stream_name}_config.yaml")
    
    # Save to YAML file
    with open(yaml_filename, "w") as file:
        yaml.dump(yaml_data, file, default_flow_style=False)

    print(f'Config file saved: {yaml_filename}')
