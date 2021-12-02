#include <iostream>
#include <vector>

constexpr uint32_t threads_per_block = 64;
constexpr uint32_t blocks_per_grid = 4096;
constexpr uint64_t max_N = 9;
constexpr size_t num_dir = 4;
using stack_word_t = uint16_t;

__device__ stack_word_t get_dir(stack_word_t word, uint32_t index) {
  return (word >> (index * 2)) & 0b11;
}

__device__ void set_dir(stack_word_t& word, uint32_t index, stack_word_t dir) {
  word &= ~(0b11 << (index * 2));
  word |= dir << (index * 2);
}

__global__ void oneesan_kernel(const uint64_t N, const uint16_t* const init_bits, const int32_t* const init_row,
    const int32_t* const init_col, uint64_t* const result, const uint64_t length) {
  constexpr uint32_t stack_size = 48;
  constexpr uint32_t stack_index_shift = 3;
  constexpr uint32_t stack_index_mask = (1 << stack_index_shift) - 1;
  constexpr size_t stack_word_size = (stack_size + stack_index_mask) >> stack_index_shift;
  stack_word_t dir_stack[stack_word_size];
  uint16_t bits[max_N];
  const uint64_t offset = threadIdx.x + blockIdx.x * blockDim.x;
  const uint64_t stride = gridDim.x * blockDim.x;
  uint64_t index = offset;
  if (index >= length) {
    result[offset] = 0;
    return;
  }
  int32_t row = init_row[index];
  int32_t col = init_col[index];
  uint32_t stack_index = 0;
  for (uint64_t i = 0; i <= N; ++i) {
    bits[i] = init_bits[index * (N + 1) + i];
  }
  bool first = true;
  uint64_t count = 0;
  int32_t dr[num_dir] = {0, 1, 0, -1};
  int32_t dc[num_dir] = {1, 0, -1, 0};
  while (true) {
    if (row == N && col == N) {
      ++count;
    } else {
      const uint32_t stack_index_high = stack_index >> stack_index_shift;
      const uint32_t stack_index_low = stack_index & stack_index_mask;
      const uint32_t dir = first ? 0 : get_dir(dir_stack[stack_index_high], stack_index_low) + 1;
      if (dir < num_dir) {
        first = false;
        set_dir(dir_stack[stack_index_high], stack_index_low, dir);
        int32_t next_row = row + dr[dir];
        int32_t next_col = col + dc[dir];
        if (min(next_row, next_col) < 0 || max(next_row, next_col) > N) {
          continue;
        }
        if ((bits[next_row] >> next_col) & 1) {
          continue;
        }
        bits[next_row] |= 1 << next_col;
        ++stack_index;
        row = next_row;
        col = next_col;
        first = true;
        continue;
      }
    }
    if (stack_index == 0) {
      index += stride;
      if (index >= length) {
        result[offset] = count;
        return;
      }
      row = init_row[index];
      col = init_col[index];
      for (uint64_t i = 0; i <= N; ++i) {
        bits[i] = init_bits[index * (N + 1) + i];
      }
      first = true;
    } else {
      bits[row] ^= 1 << col;
      --stack_index;
      const uint32_t stack_index_high = stack_index >> stack_index_shift;
      const uint32_t stack_index_low = stack_index & stack_index_mask;
      const uint32_t prev_dir = get_dir(dir_stack[stack_index_high], stack_index_low);
      row -= dr[prev_dir];
      col -= dc[prev_dir];
      first = false;
    }
  }
}

size_t expand_recursive(std::vector<uint16_t>& bits, std::vector<uint16_t>& current_bits,
    std::vector<int32_t>& rows, std::vector<int32_t>& cols, int32_t row, int32_t col, int32_t N, int32_t depth) {
  using std::begin;
  using std::end;
  if (row == N && col == N) {
    return 1;
  }
  if (depth == 0) {
    bits.insert(end(bits), begin(current_bits), end(current_bits));
    rows.push_back(row);
    cols.push_back(col);
    return 0;
  }
  int32_t dr[num_dir] = {0, 1, 0, -1};
  int32_t dc[num_dir] = {1, 0, -1, 0};
  size_t finished_count = 0;
  for (size_t dir = 0; dir < num_dir; ++dir) {
    auto next_row = row + dr[dir];
    auto next_col = col + dc[dir];
    if (std::min(next_row, next_col) < 0 || std::max(next_row, next_col) > N) continue;
    if ((current_bits[next_row] >> next_col) & 1) continue;
    current_bits[next_row] |= 1 << next_col;
    finished_count += expand_recursive(bits, current_bits, rows, cols, next_row, next_col, N, depth-1);
    current_bits[next_row] ^= 1 << next_col;
  }
  return finished_count;
}

#define HANDLE_ERROR(expr)                                                                \
  do {                                                                                    \
    auto err = expr;                                                                      \
    if (err != cudaSuccess) {                                                             \
      std::cerr << "At: " << #expr << ": " << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << std::endl; \
    }                                                                                     \
  } while (false)

int main(int argc, char** argv) {
  int32_t N = std::stoi(argv[1]);
  int32_t expand = std::stoi(argv[2]);
  std::vector<uint16_t> bits;
  std::vector<uint16_t> current_bits(N+1, 0);
  current_bits[0] |= 1;
  std::vector<int32_t> rows, cols;
  uint64_t result = expand_recursive(bits, current_bits, rows, cols, 0, 0, N, expand);
  std::cerr << bits.size() << " " << rows.size() << std::endl;
  size_t children_count = rows.size();
  const size_t threads_per_grid = blocks_per_grid * threads_per_block;
  std::vector<uint64_t> results(threads_per_grid);
  uint16_t* bits_dev = nullptr;
  int32_t* rows_dev = nullptr;
  int32_t* cols_dev = nullptr;
  uint64_t* results_dev = nullptr;
  HANDLE_ERROR(cudaMalloc(&bits_dev, bits.size() * sizeof(uint16_t)));
  HANDLE_ERROR(cudaMalloc(&rows_dev, rows.size() * sizeof(int32_t)));
  HANDLE_ERROR(cudaMalloc(&cols_dev, cols.size() * sizeof(int32_t)));
  HANDLE_ERROR(cudaMalloc(&results_dev, results.size() * sizeof(uint64_t)));
  HANDLE_ERROR(cudaMemcpy(bits_dev, bits.data(), bits.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(rows_dev, rows.data(), rows.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(cols_dev, cols.data(), cols.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
  oneesan_kernel<<<blocks_per_grid, threads_per_block>>>(N, bits_dev, rows_dev, cols_dev, results_dev, children_count);
  HANDLE_ERROR(cudaMemcpy(results.data(), results_dev, results.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost));
  for (auto&& num : results) {
    result += num;
  }
  std::cout << result << std::endl;
}
