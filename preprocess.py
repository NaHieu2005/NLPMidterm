import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.nn.utils.rnn import pad_sequence

# ======================================================================================
# SECTION 1: CẤU HÌNH (CONFIGURATION)
# Thay đổi các đường dẫn và tham số này nếu cần
# ======================================================================================

# Đường dẫn đến thư mục chứa dữ liệu
DATA_DIR = "corpus"  # Giả sử bạn chạy file code từ thư mục cha của "Released Corpus"

# Các file dữ liệu
TRAIN_EN_FILE = os.path.join(DATA_DIR, "train.en.txt")
TRAIN_VI_FILE = os.path.join(DATA_DIR, "train.vi.txt")
TEST_EN_FILE = os.path.join(DATA_DIR, "test.en.txt")
TEST_VI_FILE = os.path.join(DATA_DIR, "test.vi.txt")

# Tham số cho tokenizer
VOCAB_SIZE = 30000  # Kích thước bộ từ vựng
TOKENIZER_EN_PATH = "tokenizer_en.json"
TOKENIZER_VI_PATH = "tokenizer_vi.json"

# Các token đặc biệt
UNK_TOKEN = "[UNK]"
PAD_TOKEN = "[PAD]"
SOS_TOKEN = "[SOS]"  # Start of Sentence
EOS_TOKEN = "[EOS]"  # End of Sentence
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]

# Tham số cho DataLoader
BATCH_SIZE = 32


# ======================================================================================
# SECTION 2: ĐỌC VÀ CHIA DỮ LIỆU (LOAD AND SPLIT DATA)
# ======================================================================================
def load_and_split_data():
    """Đọc dữ liệu từ file, ghép cặp và chia thành tập train/val/test."""
    print("Bắt đầu đọc dữ liệu...")

    with open(TRAIN_EN_FILE, 'r', encoding='utf-8') as f:
        train_en = [line.strip() for line in f.readlines()]
    with open(TRAIN_VI_FILE, 'r', encoding='utf-8') as f:
        train_vi = [line.strip() for line in f.readlines()]

    with open(TEST_EN_FILE, 'r', encoding='utf-8') as f:
        test_en = [line.strip() for line in f.readlines()]
    with open(TEST_VI_FILE, 'r', encoding='utf-8') as f:
        test_vi = [line.strip() for line in f.readlines()]

    # Ghép cặp câu
    full_train_pairs = list(zip(train_en, train_vi))
    test_pairs = list(zip(test_en, test_vi))

    # Chia tập train thành train và validation (ví dụ: 90% train, 10% val)
    train_pairs, val_pairs = train_test_split(full_train_pairs, test_size=0.1, random_state=42)

    print(f"Đã đọc xong. Kích thước các tập dữ liệu:")
    print(f" - Tập huấn luyện (Train): {len(train_pairs)} cặp câu")
    print(f" - Tập xác thực (Validation): {len(val_pairs)} cặp câu")
    print(f" - Tập kiểm tra (Test): {len(test_pairs)} cặp câu")

    return train_pairs, val_pairs, test_pairs


# ======================================================================================
# SECTION 3: HUẤN LUYỆN TOKENIZER (TRAIN TOKENIZER)
# ======================================================================================
def train_tokenizer(data_pairs, lang, tokenizer_path):
    """Huấn luyện tokenizer BPE nếu file tokenizer chưa tồn tại."""
    if os.path.exists(tokenizer_path):
        print(f"Tokenizer cho ngôn ngữ '{lang}' đã tồn tại tại {tokenizer_path}. Bỏ qua huấn luyện.")
        return

    print(f"Bắt đầu huấn luyện tokenizer cho ngôn ngữ '{lang}'...")

    # 1. Khởi tạo tokenizer
    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))

    # 2. Thiết lập pre-tokenizer (tách câu thành các từ dựa trên khoảng trắng)
    tokenizer.pre_tokenizer = Whitespace()

    # 3. Thiết lập trainer
    trainer = BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=SPECIAL_TOKENS)

    # 4. Chuẩn bị dữ liệu huấn luyện (danh sách các câu)
    if lang == 'en':
        corpus = [pair[0] for pair in data_pairs]
    else:  # lang == 'vi'
        corpus = [pair[1] for pair in data_pairs]

    # 5. Huấn luyện
    tokenizer.train_from_iterator(corpus, trainer)

    # 6. Lưu tokenizer
    tokenizer.save(tokenizer_path)
    print(f"Đã lưu tokenizer cho ngôn ngữ '{lang}' tại {tokenizer_path}")


# ======================================================================================
# SECTION 4: TẠO PYTORCH DATASET (CREATE PYTORCH DATASET)
# ======================================================================================
class TranslationDataset(Dataset):
    """Lớp Dataset để cung cấp dữ liệu cho mô hình."""

    def __init__(self, data_pairs, tokenizer_en, tokenizer_vi):
        self.data_pairs = data_pairs
        self.tokenizer_en = tokenizer_en
        self.tokenizer_vi = tokenizer_vi
        self.sos_token_id_en = self.tokenizer_en.token_to_id(SOS_TOKEN)
        self.eos_token_id_en = self.tokenizer_en.token_to_id(EOS_TOKEN)
        self.sos_token_id_vi = self.tokenizer_vi.token_to_id(SOS_TOKEN)
        self.eos_token_id_vi = self.tokenizer_vi.token_to_id(EOS_TOKEN)

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        en_text, vi_text = self.data_pairs[idx]

        # Tokenize và chuyển thành ID
        en_ids = self.tokenizer_en.encode(en_text).ids
        vi_ids = self.tokenizer_vi.encode(vi_text).ids

        # Thêm token SOS và EOS
        src_ids = [self.sos_token_id_en] + en_ids + [self.eos_token_id_en]
        tgt_ids = [self.sos_token_id_vi] + vi_ids + [self.eos_token_id_vi]

        return {
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_ids, dtype=torch.long)
        }


# ======================================================================================
# SECTION 5: TẠO HÀM COLLATE (CREATE COLLATE FUNCTION)
# ======================================================================================
def create_collate_fn(pad_token_id):
    """Tạo ra hàm collate để đệm câu trong một batch."""

    def collate_fn(batch):
        src_batch, tgt_batch = [], []
        for item in batch:
            src_batch.append(item["src_ids"])
            tgt_batch.append(item["tgt_ids"])

        # Sử dụng pad_sequence để đệm các câu trong batch
        src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_token_id)
        tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_token_id)

        return {
            "src_ids": src_padded,
            "tgt_ids": tgt_padded
        }

    return collate_fn


# ======================================================================================
# SECTION 6: CHƯƠNG TRÌNH CHÍNH (MAIN EXECUTION)
# ======================================================================================
if __name__ == "__main__":
    # Bước 1: Đọc và chia dữ liệu
    train_pairs, val_pairs, test_pairs = load_and_split_data()

    # Bước 2: Huấn luyện tokenizer (chỉ dùng tập train)
    train_tokenizer(train_pairs, 'en', TOKENIZER_EN_PATH)
    train_tokenizer(train_pairs, 'vi', TOKENIZER_VI_PATH)

    # Bước 3: Tải các tokenizer đã được huấn luyện
    tokenizer_en = Tokenizer.from_file(TOKENIZER_EN_PATH)
    tokenizer_vi = Tokenizer.from_file(TOKENIZER_VI_PATH)

    pad_token_id = tokenizer_en.token_to_id(PAD_TOKEN)  # ID của token PAD là như nhau ở cả 2 tokenizer

    # Bước 4: Tạo các đối tượng Dataset
    train_dataset = TranslationDataset(train_pairs, tokenizer_en, tokenizer_vi)
    val_dataset = TranslationDataset(val_pairs, tokenizer_en, tokenizer_vi)
    test_dataset = TranslationDataset(test_pairs, tokenizer_en, tokenizer_vi)

    # Bước 5: Tạo hàm collate
    collate_fn = create_collate_fn(pad_token_id)

    # Bước 6: Tạo các đối tượng DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    print("\n========================================================")
    print("HOÀN TẤT XỬ LÝ DỮ LIỆU!")
    print(f"ID của token PAD: {pad_token_id}")
    print(f"Số lượng batch trong train_loader: {len(train_loader)}")
    print("Dữ liệu của bạn đã sẵn sàng để đưa vào mô hình.")
    print("========================================================")

    # Thử lấy một batch từ train_loader để kiểm tra
    print("\nKiểm tra một batch dữ liệu đầu ra từ train_loader:")
    try:
        one_batch = next(iter(train_loader))
        src_tensor = one_batch['src_ids']
        tgt_tensor = one_batch['tgt_ids']

        print(f" - Khóa trong batch: {list(one_batch.keys())}")
        print(f" - Kích thước tensor câu nguồn (src_ids): {src_tensor.shape}")
        print(f" - Kích thước tensor câu đích (tgt_ids): {tgt_tensor.shape}")
        print(" -> Định dạng: [Batch Size, Sequence Length]")

        print("\nVí dụ câu nguồn (dưới dạng ID) đã được đệm:")
        print(src_tensor[0])  # In ra tensor của câu đầu tiên trong batch
    except StopIteration:
        print("Lỗi: train_loader rỗng. Có thể do dữ liệu huấn luyện quá ít.")