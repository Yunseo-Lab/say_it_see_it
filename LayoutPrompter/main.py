import os
from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI

from src.preprocess import create_processor
from src.utilities import RAW_DATA_PATH, read_pt, write_pt, read_json
from src.selection import create_selector
from src.serialization import create_serializer, build_prompt
from src.parsing import Parser
from src.ranker import Ranker
from src.visualization import Visualizer, create_image_grid


class TextToLayoutPipeline:
    def __init__(
        self,
        dataset="webui",
        task="text",
        input_format="seq",
        output_format="html",
        add_unk_token=False,
        add_index_token=False,
        add_sep_token=True,
        candidate_size=-1,
        num_prompt=10,
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=1200,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        num_return=10,
        stop_token="\n\n",
    ):
        load_dotenv()
        self.dataset = dataset
        self.task = task
        self.input_format = input_format
        self.output_format = output_format
        self.add_unk_token = add_unk_token
        self.add_index_token = add_index_token
        self.add_sep_token = add_sep_token
        self.candidate_size = candidate_size
        self.num_prompt = num_prompt
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.num_return = num_return
        self.stop_token = stop_token

        self.processor = create_processor(dataset, task)
        self.serializer = create_serializer(
            dataset, task, input_format, output_format,
            add_index_token, add_sep_token, add_unk_token
        )
        self.parser = Parser(dataset=dataset, output_format=output_format)
        self.ranker = Ranker()
        self.visualizer = Visualizer(dataset)
        self.client = OpenAI()

    def get_processed_data(self, split):
        # base_dir = os.path.dirname(os.getcwd())
        base_dir = os.path.dirname(os.path.abspath(__file__))  # main.py 위치 기준
        filename = os.path.join(
            base_dir, "dataset", self.dataset, "processed", self.task, f"{split}.pt"
        )
        if os.path.exists(filename):
            return read_pt(filename, map_location="cpu")
        data = []
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        # raw_path = os.path.join(RAW_DATA_PATH(self.dataset), f"{split}.json")
        raw_path = os.path.join(base_dir, "dataset", self.dataset, f"{split}.json")
        raw_data = read_json(raw_path)
        for rd in tqdm(raw_data, desc=f"{split} data processing..."):
            data.append(self.processor(rd))
        write_pt(filename, data)
        return data

    def select_exemplars(self, train_data, test_item):
        selector = create_selector(
            task=self.task,
            train_data=train_data,
            candidate_size=self.candidate_size,
            num_prompt=self.num_prompt
        )
        return selector(test_item)

    def build_prompt(self, exemplars, test_item):
        return build_prompt(
            self.serializer, exemplars, test_item, self.dataset
        )

    def call_model(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            n=self.num_return,
        )
        return response

    def parse_response(self, response):
        return self.parser(response)

    def rank_layouts(self, parsed):
        return self.ranker(parsed)

    def visualize(self, ranked):
        images = self.visualizer(ranked)
        grid_img = create_image_grid(images)
        grid_img.save("output_grid.png")

    def run(self, test_idx=0):
        train = self.get_processed_data("train")
        _ = self.get_processed_data("val")
        test = self.get_processed_data("test")
        exemplars = self.select_exemplars(train, test[test_idx])
        prompt = self.build_prompt(exemplars, test[test_idx])
        response = self.call_model(prompt)
        parsed = self.parse_response(response)
        ranked = self.rank_layouts(parsed)
        self.visualize(ranked)


if __name__ == "__main__":
    pipeline = TextToLayoutPipeline()
    pipeline.run(test_idx=0)