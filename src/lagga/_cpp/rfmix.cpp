#include "rfmix.hpp"
#include <pybind11/stl.h>

#include <array>
#include <cassert>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

struct CRFPoint {
  std::string chrom;
  uint32_t spos;
  uint32_t epos;
  std::vector<uint8_t> ancestries;

  static CRFPoint from_msp_line(const std::string &line, size_t n_hap) {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string field;

    while (std::getline(ss, field, '\t')) {
      fields.push_back(field);
    }

    if (fields.size() < 7) {
      throw std::runtime_error("Invalid MSP line: too few fields");
    }

    CRFPoint crf;
    crf.chrom = fields[0];
    crf.spos = std::stoul(fields[1]);
    crf.epos = std::stoul(fields[2]);

    for (size_t i = 6; i < fields.size(); ++i) {
      crf.ancestries.push_back(static_cast<uint8_t>(std::stoi(fields[i])));
    }

    if (crf.ancestries.size() != n_hap) {
      throw std::runtime_error("Mismatch in number of haplotypes");
    }
    return crf;
  }
};

struct AncestryTract {
  std::string chrom;
  uint32_t spos;
  uint32_t epos;
  uint8_t anc0;
  uint8_t anc1;
};

py::dict read_rfmix(const std::string &msp_file) {
  std::ifstream infile(msp_file);
  if (!infile.is_open()) {
    throw std::runtime_error("Failed to open input file: " + msp_file);
  }

  std::string line;
  std::getline(infile, line); // skip population codes
  std::getline(infile, line); // header

  std::vector<std::string> hdr_fields;
  std::stringstream ss(line);
  std::string field;
  while (std::getline(ss, field, '\t')) {
    hdr_fields.push_back(field);
  }

  std::vector<std::pair<std::string, size_t>> hap_index_to_sample;
  for (size_t i = 6; i < hdr_fields.size(); ++i) {
    const auto &hap_field = hdr_fields[i];
    size_t dot_pos = hap_field.rfind('.');
    if (dot_pos == std::string::npos) {
      throw std::runtime_error("Invalid haplotype field: " + hap_field);
    }
    std::string sample = hap_field.substr(0, dot_pos);
    size_t hap_idx = std::stoul(hap_field.substr(dot_pos + 1));
    hap_index_to_sample.emplace_back(sample, hap_idx);
  }

  size_t n_hap = hap_index_to_sample.size();
  if (n_hap % 2 != 0) {
    throw std::runtime_error("Number of haplotypes must be even");
  }

  std::unordered_map<std::string, std::vector<AncestryTract>> sample_tracts;
  std::vector<uint8_t> prev_anc(n_hap, 0);
  std::vector<uint32_t> prev_spos(n_hap, 0);
  uint32_t cur_epos = 0;
  std::string cur_chrom = "chr0";
  bool is_first_crf = true;

  while (std::getline(infile, line)) {
    if (line.empty())
      continue;
    CRFPoint crf = CRFPoint::from_msp_line(line, n_hap);

    if (crf.chrom != cur_chrom && !is_first_crf) {
      for (size_t i = 0; i < n_hap; i += 2) {
        const auto &sample_name = hap_index_to_sample[i].first;
        sample_tracts[sample_name].push_back({cur_chrom, prev_spos[i],
                                              crf.spos - 1, prev_anc[i],
                                              prev_anc[i + 1]});
      }
      prev_spos.assign(n_hap, crf.spos);
      prev_anc = crf.ancestries;
      cur_chrom = crf.chrom;
    }

    cur_epos = crf.epos;

    if (is_first_crf) {
      is_first_crf = false;
      prev_anc = crf.ancestries;
      prev_spos.assign(n_hap, crf.spos);
      cur_chrom = crf.chrom;
    } else {
      for (size_t i = 0; i < n_hap; i += 2) {
        if (crf.ancestries[i] != prev_anc[i] ||
            crf.ancestries[i + 1] != prev_anc[i + 1]) {
          const auto &sample_name = hap_index_to_sample[i].first;
          sample_tracts[sample_name].push_back({cur_chrom, prev_spos[i],
                                                crf.spos - 1, prev_anc[i],
                                                prev_anc[i + 1]});
          prev_spos[i] = prev_spos[i + 1] = crf.spos;
          prev_anc[i] = crf.ancestries[i];
          prev_anc[i + 1] = crf.ancestries[i + 1];
        }
      }
    }
  }

  for (size_t i = 0; i < n_hap; i += 2) {
    const auto &sample_name = hap_index_to_sample[i].first;
    sample_tracts[sample_name].push_back(
        {cur_chrom, prev_spos[i], cur_epos, prev_anc[i], prev_anc[i + 1]});
  }

  py::dict result;

  std::vector<std::string> samples, chroms;
  std::vector<uint32_t> spos_vec, epos_vec;
  std::vector<int> anc0_vec, anc1_vec;

  for (const auto &[sample, tracts] : sample_tracts) {
    for (const auto &tract : tracts) {
      samples.push_back(sample);
      chroms.push_back(tract.chrom);
      spos_vec.push_back(tract.spos);
      epos_vec.push_back(tract.epos);
      anc0_vec.push_back(static_cast<int>(tract.anc0));
      anc1_vec.push_back(static_cast<int>(tract.anc1));
    }
  }

  result["sample"] = samples;
  result["chrom"] = chroms;
  result["spos"] = spos_vec;
  result["epos"] = epos_vec;
  result["anc0"] = anc0_vec;
  result["anc1"] = anc1_vec;

  return result;
}

void bind_rfmix(py::module_ &m) {
  m.def("read_rfmix", &read_rfmix,
        "Read RFMix msp file and return ancestry tracts");
}
