import galstreams
import pandas as pd
import yaml
import os


def generate(stream_name, column_name, dir=None):
    ## Import Milky Way streams from galstreams catalog and make summary
    mws = galstreams.MWStreams(verbose=False, implement_Off=False)

    # User specifies the StreamName (filter based on the Name column)

    column_name = f'{column_name}'
    stream_name = f'{stream_name}'


    # Convert to DataFrame
    df = mws.summary

    # Filter the DataFrame for the specific stream
    filtered_df = df[df[column_name] == stream_name]
    # Check if there is data after filtering
    if filtered_df.empty:
        print(f"Progenitor not found for {column_name} = {stream_name}.\nTry another column_name, name or custom coordinate.")
    else:
        # Convert filtered DataFrame to a list of dictionaries
        yaml_data = filtered_df.to_dict(orient="records")

        # Create the YAML filename based on the stream value
        config_dir = "config"
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)

        yaml_filename = os.path.join(config_dir, f"galstreams_{stream_name}_config.yaml")
        
        # Save to YAML file
        with open(yaml_filename, "w") as file:
            yaml.dump(yaml_data, file, default_flow_style=False)

        print(f'Config file saved: {yaml_filename}')


# stream_name = sys.argv[1] if len(sys.argv) > 1 else None
# column_name = sys.argv[1] if len(sys.argv) > 1 else None
# if not stream_name:
#     raise ValueError("name_o (Stream name or track name) must be specified.")
# if not stream_name:
#     raise ValueError("name_o (Stream name or track name) must be specified.")


# ## Import Milky Way streams from galstreams catalog and make summary
# mws = galstreams.MWStreams(verbose=False, implement_Off=False)

# # User specifies the StreamName (filter based on the Name column)
# column = f'{column_name}'
# stream = f'{name_o}'
