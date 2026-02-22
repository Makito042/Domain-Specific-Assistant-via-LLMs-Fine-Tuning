import pandas as pd
import json
import random

def create_conversational_dataset(input_csv, output_jsonl):
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: {input_csv} not found.")
        return

    df.columns = [c.strip() for c in df.columns]

    plant_data = {}
    for _, row in df.iterrows():
        source = str(row['Source Node']).strip().lower()
        relation = str(row['Link']).strip().lower()
        target = str(row['Destination Node']).strip().lower()

        if source not in plant_data:
            plant_data[source] = {'helps': set(), 'helped_by': set()}
        if target not in plant_data:
            plant_data[target] = {'helps': set(), 'helped_by': set()}

        if relation == 'helps':
            plant_data[source]['helps'].add(target)
            plant_data[target]['helped_by'].add(source)
        elif relation == 'helped_by':
            plant_data[source]['helped_by'].add(target)
            plant_data[target]['helps'].add(source)

    # Conversational templates
    templates_helped_by = [
        "What are the best companion plants for {plant}?",
        "What should I plant next to {plant}?",
        "Tell me some good neighbors for {plant} in my garden.",
        "Can you suggest what grows well with {plant}?",
        "I want to plant {plant}. What other plants will help it grow?"
    ]
    
    templates_helps = [
        "What plants does {plant} help grow?",
        "If I plant {plant}, what other crops will benefit from it?",
        "What does {plant} act as a good companion for?",
        "Name the plants that get a boost from being planted near {plant}."
    ]

    response_starts = [
         "The best companions for {plant} are ",
         "You should consider planting {plant} next to ",
         "{plant} grows really well alongside ",
         "Great choices to plant near {plant} include "
    ]
    
    response_helps_starts = [
         "{plant} is a great neighbor for ",
         "Planting {plant} will benefit ",
         "The following plants get a boost from {plant}: "
    ]

    dataset_data = []

    for plant, info in plant_data.items():
        plant_formatted = plant.title()

        if info['helps']:
            helps_list = sorted(list(info['helps']))
            helps_str = ", ".join(helps_list)
            
            # Generate 3 variations per plant
            for _ in range(3):
                instruction = random.choice(templates_helps).format(plant=plant_formatted)
                resp_start = random.choice(response_helps_starts).format(plant=plant_formatted)
                response = f"{resp_start}{helps_str}."
                
                dataset_data.append({
                    "instruction": instruction,
                    "input": "",
                    "output": response
                })

        if info['helped_by']:
            helped_by_list = sorted(list(info['helped_by']))
            helped_by_str = ", ".join(helped_by_list)
            
            # Generate 3 variations per plant
            for _ in range(3):
                instruction = random.choice(templates_helped_by).format(plant=plant_formatted)
                resp_start = random.choice(response_starts).format(plant=plant_formatted)
                response = f"{resp_start}{helped_by_str}."
                
                dataset_data.append({
                    "instruction": instruction,
                    "input": "",
                    "output": response
                })

    print(f"Generated {len(dataset_data)} conversational training examples.")

    with open(output_jsonl, 'w') as f:
        for item in dataset_data:
            f.write(json.dumps(item) + '\n')
            
    print(f"Saved to {output_jsonl}")

if __name__ == "__main__":
    create_conversational_dataset("companion_plants.csv", "train_conversational.jsonl")
