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
import torch


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
        # test = self.get_processed_data("test")
        # test = [{'text': 'A page for introducing transferring money at excellent exchange rates to users. The page should include one title and a background image. At the bottom, a button is needed for users to click for further information about International transfers.', 
        #         'embedding': torch.tensor([[-2.6596e-02,  1.8982e-02, -4.6906e-02,  1.2245e-02,  7.4081e-03,
        #                                     4.9713e-02,  8.1635e-03,  2.5511e-04,  6.3477e-02, -4.2328e-02,
        #                                     4.1351e-02,  9.3536e-03,  3.0823e-02,  3.0842e-03, -2.8259e-02,
        #                                     -1.9798e-03,  3.3264e-02,  3.3447e-02, -1.5478e-03,  1.3168e-02,
        #                                     1.0155e-02,  2.8934e-03,  4.8370e-03, -5.3825e-03, -1.6953e-02,
        #                                     1.8225e-03, -7.3395e-03, -2.7618e-02, -2.1011e-02,  9.1732e-05,
        #                                     2.8885e-02,  3.5797e-02, -1.4839e-02, -3.4882e-02, -3.7170e-02,
        #                                     9.2697e-03,  8.4686e-03, -6.9160e-03, -4.1556e-04, -1.3018e-03,
        #                                     -1.8311e-02, -2.7557e-02,  2.5192e-02,  2.6566e-02,  2.1759e-02,
        #                                     2.8286e-03, -1.3161e-02, -1.4029e-03, -5.1147e-02,  1.2718e-02,
        #                                     8.8654e-03,  7.0152e-03,  2.0020e-02, -9.3613e-03,  9.9277e-04,
        #                                     -3.4523e-03, -2.9129e-02, -7.9956e-03, -2.1851e-02,  1.3199e-02,
        #                                     2.6337e-02, -2.0248e-02,  3.4294e-03, -2.0355e-02, -2.0416e-02,
        #                                     -3.3142e-02, -3.3245e-03, -2.1896e-02, -1.2970e-02, -4.6463e-03,
        #                                     -1.3817e-02, -3.8513e-02, -1.2001e-02, -1.1429e-02,  2.3056e-02,
        #                                     2.7939e-02,  1.5701e-02, -1.9180e-02,  1.4145e-02,  1.9516e-02,
        #                                     5.1880e-02,  1.6136e-03, -4.8981e-03,  4.9171e-03, -9.6817e-03,
        #                                     1.1734e-02, -1.8005e-02, -8.4381e-03, -2.7557e-02, -3.3020e-02,
        #                                     -4.4495e-02, -9.7656e-03, -9.5886e-02,  3.1021e-02, -1.9165e-02,
        #                                     2.0248e-02,  2.3422e-02,  1.5030e-02, -1.9104e-02, -4.1931e-02,
        #                                     -2.1622e-02,  9.9335e-03, -4.5776e-03,  2.3575e-02,  4.8141e-03,
        #                                     -4.8096e-02, -7.4890e-02,  1.0262e-03, -6.6910e-03, -4.4495e-02,
        #                                     8.6365e-03, -2.3849e-02,  4.1443e-02,  3.5583e-02,  2.8458e-02,
        #                                     -1.1375e-02,  3.4924e-03,  2.5925e-02, -9.5291e-03, -6.5842e-03,
        #                                     -2.8205e-04,  1.3794e-02,  6.2408e-03, -3.1052e-02,  8.9264e-03,
        #                                     -6.5651e-03, -1.2299e-02, -4.7493e-03, -1.6663e-02,  1.9897e-02,
        #                                     1.5854e-02, -9.1171e-03,  2.0538e-02,  5.6592e-01, -3.0613e-03,
        #                                     -1.7487e-02, -2.9312e-02, -2.6871e-02, -2.0035e-02, -3.3836e-03,
        #                                     -3.3512e-03, -3.9673e-03, -1.7151e-02, -6.3667e-03,  7.8392e-04,
        #                                     -1.7944e-02,  2.0920e-02,  2.8458e-02, -9.2010e-03, -1.7563e-02,
        #                                     -2.5101e-02,  2.5742e-02, -2.2751e-02,  4.7226e-03, -6.6528e-03,
        #                                     1.3184e-02,  3.3508e-02, -3.8509e-03, -2.0142e-02,  4.5593e-02,
        #                                     2.5024e-02, -3.9001e-02, -2.5345e-02,  2.4471e-03, -3.5858e-02,
        #                                     2.4933e-02,  6.6711e-02,  2.2507e-02, -9.6970e-03, -7.9193e-03,
        #                                     -1.4343e-02, -1.1566e-02,  1.7899e-02, -3.2013e-02, -3.1357e-03,
        #                                     -1.6724e-02,  3.8025e-02,  1.2787e-02,  5.4016e-02, -5.6114e-03,
        #                                     -2.6215e-02, -3.6530e-02, -6.6910e-03,  1.2550e-02, -5.1422e-02,
        #                                     3.0731e-02,  3.2471e-02,  4.5662e-03, -1.1673e-02,  1.2825e-02,
        #                                     -2.7344e-02, -1.3268e-02, -2.1553e-03, -1.7014e-02,  5.7373e-03,
        #                                     -2.1896e-02,  2.0523e-02,  2.9434e-02, -3.0304e-02,  1.2924e-02,
        #                                     4.0222e-02,  1.0391e-02,  4.8518e-04,  3.0945e-02,  5.8632e-03,
        #                                     -3.6072e-02,  7.0267e-03, -8.5449e-03,  9.7961e-03, -2.4216e-02,
        #                                     -2.5116e-02, -7.0915e-03,  3.5973e-03,  5.0903e-02, -1.1040e-02,
        #                                     1.6232e-03,  3.5645e-02, -2.1439e-03,  2.7542e-02,  2.8107e-02,
        #                                     3.8757e-02, -5.1880e-02,  4.0169e-03, -2.5940e-03,  3.9886e-02,
        #                                     1.8845e-02,  1.5602e-02,  4.4250e-02, -4.7646e-03, -5.4413e-02,
        #                                     1.1475e-02,  2.0020e-02,  4.4403e-03, -4.8340e-02,  2.6642e-02,
        #                                     -8.2703e-02, -1.0887e-02,  9.1629e-03, -4.9469e-02,  3.4851e-02,
        #                                     4.7455e-02,  1.6678e-02,  8.2703e-03, -3.0136e-02, -3.9337e-02,
        #                                     1.0178e-02, -1.6678e-02,  3.6793e-03,  9.8801e-03, -2.3605e-02,
        #                                     -1.3466e-02,  2.2491e-02,  2.8015e-02,  4.8981e-03,  6.3095e-03,
        #                                     -4.8676e-02, -4.0649e-02,  3.5828e-02, -1.2772e-02, -1.3405e-02,
        #                                     -7.1526e-03, -2.0142e-02, -2.0950e-02,  2.4750e-02,  1.0767e-03,
        #                                     -2.2644e-02,  1.6388e-02, -5.1483e-02, -8.5831e-04, -7.7019e-03,
        #                                     -6.3095e-03,  1.9409e-02, -5.7411e-03,  1.5497e-03, -2.8641e-02,
        #                                     1.0468e-02, -1.1223e-02, -8.1329e-03, -1.2650e-02, -9.4757e-03,
        #                                     -3.7933e-02,  3.0060e-02, -2.2949e-02,  5.5695e-03,  2.5513e-02,
        #                                     3.0853e-02, -1.1734e-02,  8.2245e-03, -1.1063e-03, -6.9189e-04,
        #                                     -7.9422e-03, -9.6436e-03,  1.8646e-02,  1.2047e-02, -3.4542e-03,
        #                                     8.1444e-04, -3.5973e-03,  1.1192e-02,  7.1602e-03,  1.9913e-02,
        #                                     4.4861e-03,  1.7929e-02, -1.3062e-02, -1.9226e-02, -1.6129e-02,
        #                                     -1.1963e-02,  1.8036e-02,  3.7201e-02, -2.6245e-02, -2.1774e-02,
        #                                     -1.1627e-02,  2.3560e-02,  5.6641e-01, -5.1737e-05, -5.6801e-03,
        #                                     7.3792e-02,  7.6752e-03, -2.6474e-03, -3.9337e-02,  1.3027e-03,
        #                                     -1.7487e-02,  9.4223e-03,  2.8656e-02, -2.4216e-02, -1.6296e-02,
        #                                     -1.1047e-02,  1.5930e-02, -1.1749e-02, -1.7441e-02, -1.6321e-01,
        #                                     3.9337e-02, -4.5509e-03,  3.3905e-02, -1.4130e-02,  7.6447e-03,
        #                                     1.3641e-02, -2.6245e-02, -2.7752e-04, -1.8988e-03,  3.0060e-02,
        #                                     4.1779e-02,  2.3849e-02,  7.7438e-03,  4.2297e-02, -2.0123e-03,
        #                                     -9.6207e-03, -3.0632e-03,  2.4139e-02,  6.2012e-02, -2.0950e-02,
        #                                     -1.8600e-02,  2.0447e-02,  1.3504e-02,  2.6733e-02, -2.4292e-02,
        #                                     -3.8330e-02, -9.0866e-03, -1.9073e-02, -2.2430e-02, -1.5076e-02,
        #                                     3.3813e-02,  6.4812e-03,  1.5305e-02,  7.7629e-03, -1.0138e-01,
        #                                     -3.7170e-02, -2.5467e-02,  4.7882e-02, -8.2932e-03, -1.2077e-02,
        #                                     1.4229e-02,  1.6617e-02, -1.0193e-02, -1.3382e-02,  9.8724e-03,
        #                                     1.3695e-02,  3.7811e-02,  2.3026e-02, -5.0781e-02, -1.0590e-02,
        #                                     3.2926e-04,  1.8024e-03, -8.5640e-04, -3.7079e-02,  2.4063e-02,
        #                                     -2.6840e-02, -1.3041e-04, -1.7786e-03, -1.4427e-02,  3.3661e-02,
        #                                     -3.3360e-03, -4.6616e-03, -1.5053e-02,  3.0075e-02,  4.2511e-02,
        #                                     3.3051e-02, -7.6389e-04, -2.1759e-02, -3.8635e-02,  2.2751e-02,
        #                                     8.7128e-03, -8.8882e-03,  4.3716e-03,  2.1820e-02, -2.0065e-02,
        #                                     -1.8349e-03,  3.6224e-02, -1.5625e-02,  2.9587e-02, -7.2136e-03,
        #                                     -2.1534e-03, -1.3779e-02, -1.7792e-02,  3.9246e-02, -4.9934e-03,
        #                                     -3.9703e-02, -4.0131e-03,  2.7294e-03, -2.2934e-02, -2.1255e-02,
        #                                     2.6588e-03, -5.5962e-03,  7.3853e-02, -2.1393e-02,  2.5009e-02,
        #                                     -2.3483e-02,  4.4159e-02,  1.4015e-02, -6.9458e-02, -9.7179e-04,
        #                                     1.5808e-02,  5.9021e-02, -1.1978e-02, -1.0376e-02,  3.0472e-02,
        #                                     -4.7943e-02, -5.7831e-03, -3.8185e-03,  1.9302e-02,  6.9733e-03,
        #                                     3.4332e-03, -3.2654e-02,  6.3467e-04, -4.8218e-02,  3.5534e-03,
        #                                     -1.8158e-02, -2.2873e-02, -2.0386e-02, -2.1683e-02,  3.6224e-02,
        #                                     -2.3361e-02,  3.4851e-02,  2.6941e-04,  1.7014e-02, -3.1799e-02,
        #                                     -1.5430e-03,  2.2903e-02,  2.5436e-02, -3.5522e-02, -9.3613e-03,
        #                                     -4.9591e-04, -3.7937e-03, -1.4389e-02, -1.5430e-03,  1.2993e-02,
        #                                     4.0436e-02,  2.8152e-02, -3.1219e-02, -1.2589e-03, -1.3634e-02,
        #                                     -3.9024e-03,  2.1576e-02,  2.8152e-02, -8.3008e-03,  9.0332e-03,
        #                                     -1.8066e-02,  1.1421e-02, -1.9516e-02,  2.4597e-02,  3.3356e-02,
        #                                     6.3248e-03, -1.2323e-01,  1.9928e-02, -2.6840e-02,  2.3880e-02,
        #                                     1.3184e-02,  6.5727e-03,  6.9847e-03,  2.7390e-02,  4.3945e-02,
        #                                     4.3823e-02,  2.8801e-03, -2.4612e-02,  2.3300e-02,  5.1880e-03,
        #                                     -1.6968e-02,  2.4242e-03,  7.1716e-03, -1.8509e-02,  2.4582e-02,
        #                                     2.2141e-02, -1.2123e-02,  2.0645e-02,  1.1009e-02,  5.1941e-02,
        #                                     1.1185e-02, -3.6285e-02,  4.9095e-03,  3.0384e-03, -6.8359e-02,
        #                                     2.3911e-02,  1.2810e-02]], dtype=torch.float16)
        #         }]
        text = "I'm going to put a title in the top left corner and a short caption below it, and both the title and caption should only occupy the left half of the screen. The name of the organization should be placed in the bottom right corner."
        test = [self.processor(text)]

        exemplars = self.select_exemplars(train, test[test_idx])
        prompt = self.build_prompt(exemplars, test[test_idx])
        response = self.call_model(prompt)
        parsed = self.parse_response(response)
        ranked = self.rank_layouts(parsed)
        self.visualize(ranked)


if __name__ == "__main__":
    pipeline = TextToLayoutPipeline()
    pipeline.run(test_idx=0)