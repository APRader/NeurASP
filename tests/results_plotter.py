import pandas as pd
import json
import matplotlib.pyplot as plt


def create_acc_graph(filename='results.jsonl', acc='downstream'):
    """
    Imports data from a JSON Lines file, groups by seed, and generates a line graph
    of validation accuracy vs. epoch, using hyperparameters for the legend.
    """
    data = []

    # --- 1. Load and Flatten Data ---
    try:
        with open(filename, 'r') as f:
            for line in f:
                record = json.loads(line)

                flat_record = {
                    'algorithm': record.get('algorithm'),
                    'epoch': record.get('epoch'),
                    'step': record.get('step'),
                    'nn_val_accuracy': record.get('card_nn_val_accuracy'),
                    'nn_train_accuracy': record.get('card_nn_train_accuracy'),
                    'downstream_val_accuracy': record.get('downstream_val_accuracy'),
                    'seed': record.get('seed'),
                    'dataset': record.get('dataset'),
                    'batch_size': record.get('batch_size'),
                    'task': record.get('task'),
                    'lr': record.get('card_lr'),
                    'weight_decay': record.get('card_weight_decay')
                }
                data.append(flat_record)

    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON on line: {e}")
        return

    df = pd.DataFrame(data)

    # Remove any rows with missing essential plotting data
    # df.dropna(subset=['epoch', 'val_accuracy', 'seed', 'batch_size', 'lr', 'weight_decay'], inplace=True)

    if df.empty:
        print("Error: DataFrame is empty after filtering for required columns.")
        return

    # --- 2. Prepare Plotting Parameters ---

    # Get the title information from the first valid record
    algorithm = df['algorithm'].iloc[0]
    task = df['task'].iloc[0]

    plot_title = f"{algorithm} with {task}"

    # --- 3. Generate the Line Graph ---

    plt.figure(figsize=(10, 6))

    # Group the DataFrame by the unique 'seed' values
    grouped = df.groupby('seed')

    for seed, group in grouped:
        # Get hyperparameters for the legend label (assuming they are constant per seed)
        batch_size = group['batch_size'].iloc[0]
        lr = group['lr'].iloc[0]
        weight_decay = group['weight_decay'].iloc[0]

        # Create the legend label
        label = f"Seed: {seed} | Batch: {batch_size} | LR: {lr:.2} | WD: {weight_decay:.2}"

        # Plot the line
        if acc=='downstream':
            plt.plot(
                group['epoch']+group['step']/max(group['step']),
                group['downstream_val_accuracy'],
                marker='',
                linestyle='-',
                label=label
            )
        elif acc=='latent':
            plt.plot(
                group['epoch'] + group['step'] / max(group['step']),
                group['nn_val_accuracy'],
                marker='',
                linestyle='-',
                label=label
            )
        else:
            print("Not a valid acc. Please specify either downstream or latent")

    # --- 4. Finalize Plot ---
    plt.title(plot_title, fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(f'{acc} accuracy', fontsize=12)

    # Place the legend outside the plot area for cleanliness
    plt.legend(title="Hyperparameters and Seed", loc='lower right')

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout(rect=(0, 0, 0.85, 1))  # Adjust plot area to make space for the legend

    # Collect all unique seeds for the filename
    unique_seeds = sorted(df['seed'].unique())
    seed_str = "_".join(map(str, unique_seeds))
    output_filename = f"{algorithm}_{task}_{acc}_seeds_{seed_str}.png"
    plt.savefig(output_filename)
    print(f"Plot saved successfully as: {output_filename}")

def create_time_graph(filename='timings.jsonl'):
    data = []

    with open(filename, 'r') as f:
        for line in f:
            record = json.loads(line)

            flat_record = {
                # 'seed': record.get('seed'),
                # 'task': record.get('task'),
                'num_models': record.get('num_models'),
                'neurasp_prob_time': record.get('neurasp_prob_time'),
                'slash_prob_time': record.get('slash_prob_time'),
                'newrasp_prob_time': record.get('newrasp_prob_time'),
                'neurasp_grad_time': record.get('neurasp_grad_time'),
                'slash_grad_time': record.get('slash_grad_time'),
                'newrasp_grad_time': record.get('newrasp_grad_time'),
            }
            data.append(flat_record)
    df = pd.DataFrame(data)
    stats = df.groupby('num_models').agg(['mean', 'std']).reset_index()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    methods = [
        ('neurasp', 'r', 'NeurASP original', 'o'),
        ('slash', 'b', 'SLASH', 'x'),
        ('newrasp', 'tab:orange', 'NeurASP improved', 'D')
    ]

    def add_plot_lines(ax, prefix, title):
        """Helper to add mean lines and std-dev shading to a specific axis."""
        for key, color, label, marker in methods:
            col_name = f'{key}_{prefix}_time'

            x = stats['num_models']
            y_mean = stats[col_name]['mean']
            y_std = stats[col_name]['std']

            # Plot the average line
            ax.plot(x, y_mean, label=label, color=color, marker=marker, linewidth=2)

            # Fill the standard deviation area
            ax.fill_between(x, y_mean - y_std, y_mean + y_std, color=color, alpha=0.15)

        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.set_xlabel('Number of answer sets', fontsize=14)
        ax.set_ylabel('Time (seconds)', fontsize=14)
        ax.set_xscale('log', base=10)
        ax.set_yscale('log')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=11)

    # Create the two subplots
    add_plot_lines(ax1, 'prob', 'Probability calculation speeds')
    add_plot_lines(ax2, 'grad', 'Gradient calculation speeds')

    # plt.suptitle(f"Performance Analysis: {df['task'].iloc[0] if not df.empty else ''}", fontsize=16)
    plt.tight_layout()

    # Save the output
    output_name = "synthetic_timings.pdf"
    plt.savefig(output_name, format='pdf')
    print(f"Plot saved as {output_name}")
    plt.show()


if __name__ == "__main__":
    # create_acc_graph("../examples/card_arithmetic/results/card_arithmetic_3p_results.jsonl")
    # create_acc_graph("../examples/card_arithmetic/results/card_arithmetic_3p_results.jsonl", acc='latent')
    create_time_graph("results/synthetic_timings.jsonl")