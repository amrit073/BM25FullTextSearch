use std::{
    collections::HashMap,
    collections::HashSet,
    env,
    fs::{self},
    io::{self, Write},
    time::Instant,
};

struct BM25<'a> {
    corpus: &'a Vec<Vec<String>>,
    k1: f32,
    b: f32,
    doc_lengths: Vec<i32>,
    avg_doc_length: i32,
    doc_count: i32,
    tf_cache: Vec<HashMap<String, i32>>,
    term_idf: HashMap<String, f32>,
}

struct Counter<T: Eq + std::hash::Hash> {
    counts: HashMap<T, i32>,
}

impl<T: Eq + std::hash::Hash> Counter<T> {
    fn new() -> Counter<T> {
        Counter {
            counts: HashMap::new(),
        }
    }

    fn increment(&mut self, item: T) {
        let count = self.counts.entry(item).or_insert(0);
        *count += 1;
    }

    fn get_counts(&self) -> &HashMap<T, i32> {
        &self.counts
    }
}

impl<'a> BM25<'a> {
    fn calculate_idf(&mut self) {
        let mut doc_freq: HashMap<&String, i32> = HashMap::new();

        for doc in self.corpus {
            let mut unique_terms = doc.iter().collect::<HashSet<_>>();
            for term in unique_terms.drain() {
                *doc_freq.entry(term).or_insert(0) += 1;
            }
        }

        for (term, &count) in &doc_freq {
            let idf = ((self.doc_count as f32 - count as f32 + 0.5) / (count as f32 + 0.5)).ln();
            self.term_idf.insert(term.to_string(), idf);
        }
    }

    fn calculate_bm25_score(&self, query: &Vec<&str>, doc_index: usize) -> f32 {
        let mut score: f32 = 0.0;
        for term in query {
            if let Some(hash) = self.tf_cache.get(doc_index) {
                let tf = hash.get(*term).unwrap_or(&0);
                let idf = self.term_idf.get(*term).unwrap_or(&0.0);
                let numerator = *tf as f32 * (self.k1 + 1.0);
                let denominator = *tf as f32
                    + self.k1
                        * (1.0 - self.b
                            + self.b
                                * (*self.doc_lengths.get(doc_index).unwrap_or(&0) as f32
                                    / self.avg_doc_length as f32));
                score += idf * (numerator / denominator);
            }
        }
        score
    }

    fn rank_documents(&self, query: Vec<&str>) -> Vec<(i32, f32)> {
        let mut ranks: Vec<(i32, f32)> = vec![];
        for i in 0..self.doc_count as usize {
            ranks.push((i as i32, self.calculate_bm25_score(&query, i)));
        }
        ranks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        ranks
    }

    fn new(corpus: &Vec<Vec<String>>) -> BM25 {
        let mut total_doc_length = 0;
        let doc_count = corpus.len() as i32;
        let doc_lengths: Vec<i32> = corpus
            .iter()
            .map(|e| {
                let a = e.len() as i32;
                total_doc_length += a;
                a
            })
            .collect();
        let avg_doc_length = total_doc_length / doc_count;
        let mut a = BM25 {
            corpus,
            k1: 1.5,
            b: 0.75,
            doc_lengths,
            avg_doc_length,
            doc_count,
            tf_cache: corpus
                .iter()
                .map(|doc| {
                    let mut doc_counter = Counter::new();
                    doc.iter().for_each(|word| {
                        doc_counter.increment(word.clone());
                    });
                    doc_counter.get_counts().to_owned()
                })
                .collect(),
            term_idf: HashMap::new(),
        };
        a.calculate_idf();
        a
    }
}

fn list_files_with_full_paths(directory_path: &str) -> std::io::Result<Vec<String>> {
    let entries = fs::read_dir(directory_path)?;

    let file_paths: Vec<String> = entries
        .filter_map(|entry| {
            entry
                .ok()
                .and_then(|e| e.path().to_str().map(|s| s.to_string()))
        })
        .collect();

    Ok(file_paths)
}

fn read_file_words(file_path: &str) -> std::io::Result<Vec<String>> {
    let file = std::fs::read_to_string(file_path)?;
    let words: Vec<String> = file
        .split_whitespace()
        .map(|s| s.to_lowercase().to_string())
        .collect();
    Ok(words)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <text_file_directory>", args[0]);
        std::process::exit(1);
    }

    let text_file_dir = &args[1];
    let mut all_files = list_files_with_full_paths(text_file_dir).unwrap();
    all_files.append(&mut all_files.clone());
    let my: Vec<Vec<String>> = all_files
        .iter()
        .map(|file_path| read_file_words(file_path).unwrap())
        .collect();
    let start_time = Instant::now();
    let ins = BM25::new(&my);
    let end_time = Instant::now();
    let duration = end_time.duration_since(start_time);
    println!(
        "Time taken to create index: {}.{:03} seconds",
        duration.as_secs(),
        duration.subsec_millis()
    );
    loop {
        print!("Enter a search query: ");
        io::stdout().flush().unwrap();
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let query: Vec<&str> = input.split_whitespace().collect();
        let ranks = ins.rank_documents(query);
        println!("Results:");
        for (index, score) in ranks.iter().take(5) {
            println!(
                "{}: BM25 Score - {}",
                all_files
                    .get(*index as usize)
                    .unwrap()
                    .split("/")
                    .last()
                    .unwrap(),
                score
            );
        }
        println!("---------------------")
    }
}
