from topicnet.cooking_machine.config_parser import (
    build_experiment_environment_from_yaml_config
)


from topicnet.cooking_machine.recipes import (
    ARTM_baseline as config_string
)

dataset_path = '/data/datasets/NIPS/dataset.csv'

specific_topics   = [f'spc_topic_{i}' for i in range(19)]
background_topics = [f'bcg_topic_{i}' for i in range( 1)]

config_string = config_string.format(
    dataset_path=dataset_path,
    modality_list=['@word'],
    main_modality='@word',
    specific_topics=specific_topics,
    background_topics=background_topics
)
experiment, dataset = (
    build_experiment_environment_from_yaml_config(
        yaml_string=config_string,
        experiment_id='sample_config',
        save_path='sample_save_path'
    )
)
experiment.run(dataset)
