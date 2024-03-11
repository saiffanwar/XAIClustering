import matplotlib.pyplot as plt
import pickle as pck
import numpy as np

def plot_results(kernel_widths=None):
    colours = ['r', 'b', 'g', 'black']
    fig, axes = plt.subplots(2, 3, figsize=(15, 11))
    plt.subplots_adjust(hspace=0.4)
    ax = fig.get_axes()
    for i, kernel_width in enumerate(kernel_widths):
        with open(f'saved/PHM08_results_0.5_0.05_5_0.5.pck', 'rb') as file:
            results = pck.load(file)
        new_results = {}
        for k,v in results.items():
            if v != []:
                new_results[k] = v
        results = new_results
        for j in range(3):
            ys = [results[instance][j] for instance in results.keys() if results[instance] != []]
            if j == 0:
                avg_llc_error = np.mean(ys)
            if j == 1:
                avg_chilli_error = np.mean(ys)
            if j ==2:
                avg_lime_error = np.mean(ys)
            if max(ys) > 1000:
                ax[i].set_yscale('log')
            xs = [x for x in range(len(ys))]
            ax[i].scatter(xs, ys, c=colours[j], s=20, marker='x')
        ax[i].vlines(xs,[results[instance][0] for instance in results.keys()], [results[instance][1] for instance in results.keys()], color='gray', label='_nolegend_')
        ax[i].set_title(r'$\sigma$ = '+str(kernel_width)+f'\n New CHILLI Average MSE: {avg_newchilli_error:.2f}, \n CHILLI Average MSE: {avg_chilli_error:.2f}, \n LIME Average MSE: {avg_lime_error:.2f}')
        ax[i].set_xlabel('Instance')
        ax[i].set_ylabel('Explanation error (MSE)')
    fig.legend(['CHILLI with automated locality', 'CHILLI without automated locality', 'LIME'],loc='center', bbox_to_anchor=(0.5,0.98), ncols=3)
    fig.savefig('Figures/Results.pdf', bbox_inches='tight')



def plot_results2():
    with open(f'saved/results/PHM08_results_0.5_0.05_5_0.5.pck', 'rb') as file:
        results = pck.load(file)
    model_predictions, llc_predictions, chilli_predictions = results[0], results[1], results[2]
    fig, axes = plt.subplots(1,1, figsize=(8, 4))
    for instance in range(len(model_predictions)):
        axes.scatter([instance], [model_predictions[instance]], c='r', s=30, marker='x')
        axes.scatter([instance], [chilli_predictions[instance]], c='b', s=20, marker='x')
        axes.scatter([instance], [llc_predictions[instance]], c='g', s=20, marker='x')

        axes.vlines([instance],[min(model_predictions[instance], chilli_predictions[instance], llc_predictions[instance])], [max(model_predictions[instance], chilli_predictions[instance], llc_predictions[instance])], color='gray', label='_nolegend_')
    fig.legend(['Model predictions', 'CHILLI predictions', 'LLC predictions'],loc='center', bbox_to_anchor=(0.5,0.98), ncols=3)
    fig.savefig('Figures/Results.pdf', bbox_inches='tight')
plot_results2()

def compare_parameters(parameter_search):
#    sparsitys = []
#    coverages = []
#    starting_ks = []
#    neighbourhoods = []
    rmses = []
    results = []
    missing_results = []
    i=0
    sparsitys = {v: [] for v in parameter_search['sparsity']}
    coverages = {v: [] for v in parameter_search['coverage']}
    starting_ks = {v: [] for v in parameter_search['starting_k']}
    neihgbourhoods = {v: [] for v in parameter_search['neighbourhood']}

    for sparsity_threshold in parameter_search['sparsity']:
        for coverage_threshold in parameter_search['coverage']:
            for starting_k in parameter_search['starting_k']:
                for neighbourhood_threshold in parameter_search['neighbourhood']:

                    if not os.path.exists(f'saved/feature_ensembles/phm08_feature_ensembles_full_{sparsity_threshold}_{coverage_threshold}_{starting_k}_{neighbourhood_threshold}.pck'):
                        missing_results.append([sparsity_threshold, coverage_threshold, starting_k, neighbourhood_threshold])
                    else:
                        with open(f'saved/results/phm08_results_{sparsity_threshold}_{coverage_threshold}_{starting_k}_{neighbourhood_threshold}.pck', 'rb') as file:
                            model_predictions, llc_predictions, rmse = pck.load(file)
                            i+=1
#                        sparsitys.append(sparsity_threshold)
#                        coverages.append(coverage_threshold)
#                        starting_ks.append(starting_k)
#                        neighbourhoods.append(neighbourhood_threshold)
                        rmses.append(rmse)
    min_rmse = min(rmses)
    max_rmse = max(rmses)

    selected_params = {v: [] for v in ['sparsity', 'coverage', 'starting_k', 'neighbourhood']}
    for fix1 in ['sparsity', 'coverage', 'starting_k', 'neighbourhood']:
        for fix2 in ['sparsity', 'coverage', 'starting_k', 'neighbourhood']:
            if fix1 != fix2:
                comparing_params = [t for t in ['sparsity', 'coverage', 'starting_k', 'neighbourhood'] if t not in [fix1, fix2]]
                fig, axes = plt.subplots(len(parameter_search[fix1]), len(parameter_search[fix2]), figsize=(20, 20))
                for i in parameter_search[fix1]:
                    selected_params[fix1] = i
                    for j in parameter_search[fix2]:
                        selected_params[fix2] = j
                        results = []
                        for k in parameter_search[comparing_params[0]]:
                            row = []
                            selected_params[comparing_params[0]] = k
                            for l in parameter_search[comparing_params[1]]:
                                selected_params[comparing_params[1]] = l
                                with open(f"saved/results/phm08_results_{selected_params['sparsity']}_{selected_params['coverage']}_{selected_params['starting_k']}_{selected_params['neighbourhood']}.pck", 'rb') as file:
                                    model_predictions, llc_predictions, rmse = pck.load(file)
                                    row.append(rmse)
                            results.append(row)
                        results = np.array(results)
                        for xpos in range(len(parameter_search[comparing_params[0]])):
                            for ypos in range(len(parameter_search[comparing_params[1]])):
                                text = axes[parameter_search[fix1].index(i), parameter_search[fix2].index(j)].text(ypos, xpos, round(results[xpos][ypos], 1),
                                               ha = "center", va = "center", color = "w")
                        im = axes[parameter_search[fix1].index(i), parameter_search[fix2].index(j)].imshow(results, vmin=min_rmse, vmax=max_rmse, cmap='viridis')
                        axes[parameter_search[fix1].index(i), parameter_search[fix2].index(j)].set_title(f'{fix1}={i}, {fix2}={j}')
                        axes[parameter_search[fix1].index(i), parameter_search[fix2].index(j)].set_xticks(range(len(parameter_search[comparing_params[1]])))
                        axes[parameter_search[fix1].index(i), parameter_search[fix2].index(j)].set_yticks(range(len(parameter_search[comparing_params[0]])))
                        axes[parameter_search[fix1].index(i), parameter_search[fix2].index(j)].set_xticklabels(parameter_search[comparing_params[1]])
                        axes[parameter_search[fix1].index(i), parameter_search[fix2].index(j)].set_yticklabels(parameter_search[comparing_params[0]])
                        axes[parameter_search[fix1].index(i), parameter_search[fix2].index(j)].set_xlabel(comparing_params[1])
                        axes[parameter_search[fix1].index(i), parameter_search[fix2].index(j)].set_ylabel(comparing_params[0])
                fig.colorbar(im, ax=axes.ravel().tolist())
                fig.savefig(f'figures/phm08/parameter_search/comparing_{comparing_params[0]}_{comparing_params[1]}.pdf')

    print(len(missing_results), i)
    with open('saved/missing_results.pck', 'wb') as file:
        pck.dump(missing_results, file)
