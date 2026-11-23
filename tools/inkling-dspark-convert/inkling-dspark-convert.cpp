// llama-inkling-dspark-convert: Inkling-2 DSpark drafter (mtp.safetensors + config.json mtp_config) -> llama.cpp "dflash" GGUF
//
//   llama-inkling-dspark-convert <mtp.safetensors> <config.json> <target.gguf> <out.gguf>
//
// The draft shares the target's token embeddings, embedding norm and output head (ctx_other), so it carries no
// vocab-sized tensors except markov_w1. Large matrices are Q8_0, norms and vectors F32, small matrices F16.
// The tokenizer metadata is copied from the target GGUF.

#include "ggml.h"
#include "gguf.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;

struct st_tensor {
    std::string        dtype;
    std::vector<int64_t> shape; // safetensors order: outermost first
    size_t             off0, off1;
};

static std::vector<uint8_t> read_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static float bf16_to_f32(uint16_t v) {
    uint32_t u = (uint32_t) v << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

int main(int argc, char ** argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s <mtp.safetensors> <config.json> <target.gguf> <out.gguf>\n", argv[0]);
        return 1;
    }
    const std::string p_st = argv[1], p_cfg = argv[2], p_tgt = argv[3], p_out = argv[4];

    // safetensors
    const std::vector<uint8_t> st = read_file(p_st);
    uint64_t hlen;
    memcpy(&hlen, st.data(), 8);
    const json hdr = json::parse(std::string((const char *) st.data() + 8, hlen));
    const uint8_t * st_data = st.data() + 8 + hlen;
    std::map<std::string, st_tensor> T;
    for (auto & [name, v] : hdr.items()) {
        if (name == "__metadata__") continue;
        st_tensor t;
        t.dtype = v["dtype"].get<std::string>();
        t.shape = v["shape"].get<std::vector<int64_t>>();
        t.off0  = v["data_offsets"][0].get<size_t>();
        t.off1  = v["data_offsets"][1].get<size_t>();
        if (t.dtype != "BF16") throw std::runtime_error(name + ": expected BF16, got " + t.dtype);
        T[name] = t;
    }

    // config
    const json cfg = json::parse(read_file(p_cfg));
    const json & mc = cfg.at("mtp_config");
    const json & tc = cfg.at("text_config");
    const int D  = mc.at("hidden_size").get<int>();
    const int L  = mc.at("num_hidden_layers").get<int>();
    const int H  = mc.at("num_attention_heads").get<int>();
    const int KV = mc.at("num_key_value_heads").get<int>();
    const int hd = mc.at("head_dim").get<int>();
    const int I  = mc.at("intermediate_size").get<int>();
    const int B  = mc.at("block_size").get<int>();
    const int R  = mc.at("markov_rank").get<int>();
    const int W  = mc.at("sliding_window_size").get<int>();
    const float eps   = mc.at("rms_norm_eps").get<float>();
    const float theta = mc.at("rope_theta").get<float>();
    const float softcap = tc.value("final_logit_softcapping", 0.0f);
    const std::vector<int> local = mc.at("local_layer_ids").get<std::vector<int>>();
    const std::vector<int> taps  = mc.at("target_layer_ids").get<std::vector<int>>();
    if (mc.at("markov_head_type").get<std::string>() != "rnn_hidden") throw std::runtime_error("markov_head_type must be rnn_hidden");
    const int nt = (int) taps.size();

    // fuse_weight [L + 1, nt]: softmax of each row (the last row is the markov head's)
    std::vector<float> alpha((L + 1) * nt);
    {
        const st_tensor & fw = T.at("fuse_weight");
        if (fw.shape != std::vector<int64_t>{L + 1, nt}) throw std::runtime_error("fuse_weight shape");
        const uint16_t * p = (const uint16_t *) (st_data + fw.off0);
        for (int i = 0; i <= L; ++i) {
            float mx = -1e30f, s = 0.0f;
            for (int j = 0; j < nt; ++j) mx = std::max(mx, bf16_to_f32(p[i*nt + j]));
            for (int j = 0; j < nt; ++j) s += (alpha[i*nt + j] = expf(bf16_to_f32(p[i*nt + j]) - mx));
            for (int j = 0; j < nt; ++j) alpha[i*nt + j] /= s;
        }
    }

    // target: tokenizer metadata and the vocab size
    gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    gguf_context * gtgt = gguf_init_from_file(p_tgt.c_str(), gp);
    if (!gtgt) throw std::runtime_error("cannot read " + p_tgt);

    gguf_context * gout = gguf_init_empty();
    gguf_set_kv(gout, gtgt);
    for (int64_t i = gguf_get_n_kv(gout) - 1; i >= 0; --i) {
        const std::string k = gguf_get_key(gout, i);
        if (k.rfind("tokenizer.", 0) != 0) gguf_remove_key(gout, k.c_str());
    }
    const int64_t kid = gguf_find_key(gout, "tokenizer.ggml.tokens");
    if (kid < 0) throw std::runtime_error("target has no tokenizer.ggml.tokens");
    const int64_t n_vocab = (int64_t) gguf_get_arr_n(gout, kid);

    const std::string a = "dflash";
    gguf_set_val_str (gout, "general.architecture", a.c_str());
    gguf_set_val_str (gout, "general.name", "Inkling-2-Mini DSpark draft");
    gguf_set_val_u32 (gout, (a + ".context_length").c_str(), mc.value("max_position_embeddings", 1048576));
    gguf_set_val_u32 (gout, (a + ".embedding_length").c_str(), D);
    gguf_set_val_u32 (gout, (a + ".block_count").c_str(), L);
    gguf_set_val_u32 (gout, (a + ".feed_forward_length").c_str(), I);
    gguf_set_val_u32 (gout, (a + ".attention.head_count").c_str(), H);
    gguf_set_val_u32 (gout, (a + ".attention.head_count_kv").c_str(), KV);
    gguf_set_val_u32 (gout, (a + ".attention.key_length").c_str(), hd);
    gguf_set_val_u32 (gout, (a + ".attention.value_length").c_str(), hd);
    gguf_set_val_f32 (gout, (a + ".attention.layer_norm_rms_epsilon").c_str(), eps);
    gguf_set_val_f32 (gout, (a + ".rope.freq_base").c_str(), theta);
    gguf_set_val_u32 (gout, (a + ".rope.dimension_count").c_str(), hd);
    gguf_set_val_u32 (gout, (a + ".attention.sliding_window").c_str(), W);
    {
        std::vector<int8_t> pat(L, 0); // GGUF bools are one byte
        for (int i : local) if (i >= 0 && i < L) pat[i] = 1;
        gguf_set_arr_data(gout, (a + ".attention.sliding_window_pattern").c_str(), GGUF_TYPE_BOOL, pat.data(), L);
    }
    gguf_set_val_f32 (gout, (a + ".final_logit_softcapping").c_str(), softcap);
    gguf_set_val_u32 (gout, (a + ".block_size").c_str(), B);
    gguf_set_arr_data(gout, (a + ".target_layers").c_str(), GGUF_TYPE_INT32, taps.data(), nt);
    gguf_set_arr_data(gout, (a + ".fuse_alpha").c_str(), GGUF_TYPE_FLOAT32, alpha.data(), (L + 1)*nt);
    gguf_set_val_bool(gout, (a + ".sample_from_anchor").c_str(), false);
    // mask rows take mask_embd in the graph whatever their id; any valid id keeps the batch well formed
    if (gguf_find_key(gout, "tokenizer.ggml.mask_token_id") < 0) {
        gguf_set_val_u32(gout, "tokenizer.ggml.mask_token_id", 0);
    }

    // tensors
    size_t mem = 64ull*1024*1024;
    for (auto & [n, t] : T) mem += (t.off1 - t.off0) * 2 + 1024; // f32 upper bound
    ggml_init_params ip = { mem, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    std::vector<float> f32;
    auto load_f32 = [&](const st_tensor & t, int64_t rows_keep) {
        int64_t ne = 1;
        for (auto d : t.shape) ne *= d;
        const int64_t row = t.shape.back();
        const int64_t n = rows_keep >= 0 ? rows_keep * row : ne;
        f32.resize(n);
        const uint16_t * p = (const uint16_t *) (st_data + t.off0);
        for (int64_t i = 0; i < n; ++i) f32[i] = bf16_to_f32(p[i]);
    };

    auto add = [&](const std::string & src, const std::string & dst, ggml_type type, int64_t rows_keep = -1) {
        const st_tensor & t = T.at(src);
        load_f32(t, rows_keep);
        ggml_tensor * out;
        if (t.shape.size() == 1) {
            out = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, t.shape[0]);
            memcpy(out->data, f32.data(), f32.size()*sizeof(float));
        } else {
            const int64_t ne0 = t.shape[1], ne1 = rows_keep >= 0 ? rows_keep : t.shape[0];
            out = ggml_new_tensor_2d(ctx, type, ne0, ne1);
            if (type == GGML_TYPE_F32) {
                memcpy(out->data, f32.data(), f32.size()*sizeof(float));
            } else if (type == GGML_TYPE_F16) {
                ggml_fp32_to_fp16_row(f32.data(), (ggml_fp16_t *) out->data, ne0*ne1);
            } else {
                ggml_quantize_chunk(type, f32.data(), out->data, 0, ne1, ne0, nullptr);
            }
        }
        ggml_set_name(out, dst.c_str());
        gguf_add_tensor(gout, out);
        fprintf(stderr, "  %-44s -> %-28s %s [%lld, %lld]\n", src.c_str(), dst.c_str(), ggml_type_name(out->type),
                (long long) out->ne[0], (long long) out->ne[1]);
    };

    const ggml_type QT = GGML_TYPE_Q8_0;
    for (int i = 0; i < L; ++i) {
        const std::string s = "layers." + std::to_string(i) + ".", d = "blk." + std::to_string(i) + ".";
        add(s + "input_layernorm.weight",          d + "attn_norm.weight",       GGML_TYPE_F32);
        add(s + "post_attention_layernorm.weight", d + "ffn_norm.weight",        GGML_TYPE_F32);
        add(s + "dflare_norm.weight",              d + "dflare_norm.weight",     GGML_TYPE_F32);
        add(s + "self_attn.q_norm.weight",         d + "attn_q_norm.weight",     GGML_TYPE_F32);
        add(s + "self_attn.k_norm.weight",         d + "attn_k_norm.weight",     GGML_TYPE_F32);
        add(s + "self_attn.k_norm_ctx.weight",     d + "attn_k_ctx_norm.weight", GGML_TYPE_F32);
        add(s + "self_attn.q_proj.weight",         d + "attn_q.weight",          QT);
        add(s + "self_attn.k_proj.weight",         d + "attn_k.weight",          QT);
        add(s + "self_attn.v_proj.weight",         d + "attn_v.weight",          QT);
        add(s + "self_attn.k_proj_ctx.weight",     d + "attn_k_ctx.weight",      QT);
        add(s + "self_attn.v_proj_ctx.weight",     d + "attn_v_ctx.weight",      QT);
        add(s + "self_attn.o_proj.weight",         d + "attn_output.weight",     QT);
        add(s + "mlp.gate_proj.weight",            d + "ffn_gate.weight",        QT);
        add(s + "mlp.up_proj.weight",              d + "ffn_up.weight",          QT);
        add(s + "mlp.down_proj.weight",            d + "ffn_down.weight",        QT);
    }
    add("norm.weight",                      "output_norm.weight",  GGML_TYPE_F32);
    add("mask_embed",                       "mask_embd.weight",    GGML_TYPE_F32);
    add("markov_head.dflare_norm.weight",   "markov_norm.weight",  GGML_TYPE_F32);
    add("markov_head.rnn_proj.weight",      "joint_proj.weight",   GGML_TYPE_F16);
    add("markov_head.rnn_proj.bias",        "joint_proj.bias",     GGML_TYPE_F32);
    add("markov_head.hidden_out.weight",    "markov_out.weight",   GGML_TYPE_F16);
    add("markov_head.hidden_out.bias",      "markov_out.bias",     GGML_TYPE_F32);
    // markov_w1 has the padded vocab's rows; keep the tokenizer's
    const int64_t w1_rows = std::min<int64_t>(n_vocab, T.at("markov_head.markov_w1.weight").shape[0]);
    add("markov_head.markov_w1.weight",     "markov_w1.weight",    QT, w1_rows);

    fprintf(stderr, "vocab %lld, taps [", (long long) n_vocab);
    for (int j = 0; j < nt; ++j) fprintf(stderr, "%s%d", j ? ", " : "", taps[j]);
    fprintf(stderr, "], block %d, rank %d, softcap %.1f\nfuse alpha:\n", B, R, softcap);
    for (int i = 0; i <= L; ++i) {
        fprintf(stderr, "  %s:", i < L ? ("layer " + std::to_string(i)).c_str() : "head   ");
        for (int j = 0; j < nt; ++j) fprintf(stderr, " %.4f", alpha[i*nt + j]);
        fprintf(stderr, "\n");
    }

    if (!gguf_write_to_file(gout, p_out.c_str(), false)) throw std::runtime_error("cannot write " + p_out);
    fprintf(stderr, "wrote %s\n", p_out.c_str());

    gguf_free(gout);
    gguf_free(gtgt);
    ggml_free(ctx);
    return 0;
}
