use std::time::Instant;
use serenade_optimized::vmisknn::vmis_index::{read_into_vecs, vecs_to_training_data};
use serenade_optimized::vmisknn::vmis_index::VMISIndex;
use serenade_optimized::vmisknn::predict;
use serenade_optimized::io::read_test_data_evolving;

use rand::Rng;
#[allow(deprecated)] use rand::XorShiftRng;
use rand::prelude::*;

fn main() {

    let num_clicks_to_delete = 7;

    for seed in [42, 789, 1234] {
        for num_active_queries in [100, 1000, 10000] {
            run_experiment(
                "ecom1m",
                "../snapcase/datasets/session/bolcom-clicks-1m_train.txt",
                "../snapcase/datasets/session/bolcom-clicks-1m_test.txt",
                num_active_queries,
                num_clicks_to_delete,
                seed,
            );

            run_experiment(
                "rsc15",
                "../snapcase/datasets/session/rsc15-clicks_train_full.txt",
                "../snapcase/datasets/session/rsc15-clicks_test.txt",
                num_active_queries,
                num_clicks_to_delete,
                seed,
            );

            run_experiment(
                "ecom60m",
                "..snapcase/datasets/session/bolcom-clicks-50m_train.txt",
                "../snapcase/datasets/session/bolcom-clicks-50m_test.txt",
                num_active_queries,
                num_clicks_to_delete,
                seed,
            );
        }
    }
}

fn run_experiment(
    dataset: &str,
    train_path: &str,
    test_path: &str,
    num_active_queries: usize,
    num_clicks_to_delete: usize,
    seed: u64
) {

    eprintln!("dataset={},num_active_queries={}", dataset, num_active_queries);

    let (mut train_session_ids, mut train_item_ids, mut train_times) = read_into_vecs(train_path);
    let test_sessions = read_test_data_evolving(test_path);

    #[allow(deprecated)] let mut rng = XorShiftRng::seed_from_u64(seed);

    let query_session_ids: Vec<_> = test_sessions.keys()
        .choose_multiple(&mut rng, num_active_queries)
        .into_iter()
        .map(|id| *id)
        .collect();


    let sampled_queries: Vec<_> = test_sessions.into_iter()
        .filter(|(id, _items)| query_session_ids.contains(id))
        .map(|(id, items)| {
            let random_session_length = rng.gen_range(1, items.len());
            (id, items[..random_session_length].to_vec())
        })
        .collect();

    for _run in 0..num_clicks_to_delete {

        let random_click_to_delete = rng.gen_range(0, train_session_ids.len());

        train_session_ids.remove(random_click_to_delete);
        train_item_ids.remove(random_click_to_delete);
        train_times.remove(random_click_to_delete);

        let cloned_session_ids = train_session_ids.clone();
        let cloned_item_ids = train_item_ids.clone();
        let cloned_times = train_times.clone();

        let start = Instant::now();

        let data_train = vecs_to_training_data(
            cloned_session_ids, cloned_item_ids, cloned_times
        ).unwrap();

        let vmis_index = VMISIndex::new_from_train_data(data_train, 500, 0.1);

        let mut dummy_for_execution = 0;

        for (_, items) in sampled_queries.iter() {
            let predictions = predict(&vmis_index, &items, 100, 500, 20, false);
            dummy_for_execution += predictions.len()
        }
        let duration = start.elapsed().as_micros();
        eprintln!("# dummy output to avoid compiling away the result: {}", dummy_for_execution);
        println!(
            "vmis_rust,deletion_performance,{},{},{}",
            dataset,
            num_active_queries,
            duration
        );
    }
}