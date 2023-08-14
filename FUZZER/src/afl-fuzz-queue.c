/*
   american fuzzy lop++ - queue relates routines
   ---------------------------------------------

   Originally written by Michal Zalewski

   Now maintained by Marc Heuse <mh@mh-sec.de>,
                        Heiko Eißfeldt <heiko.eissfeldt@hexco.de> and
                        Andrea Fioraldi <andreafioraldi@gmail.com>

   Copyright 2016, 2017 Google Inc. All rights reserved.
   Copyright 2019-2023 AFLplusplus Project. All rights reserved.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at:

     https://www.apache.org/licenses/LICENSE-2.0

   This is the real deal: the program takes an instrumented binary and
   attempts a variety of basic fuzzing tricks, paying close attention to
   how they affect the execution path.

 */

#include "afl-fuzz.h"
#include <limits.h>
#include <ctype.h>
#include <math.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <zlib.h>

#define BATCH_SIZE 4

#ifdef _STANDALONE_MODULE
void minimize_bits(afl_state_t *afl, u8 *dst, u8 *src) {

  return;

}

void run_afl_custom_queue_new_entry(afl_state_t *afl, struct queue_entry *q,
                                    u8 *a, u8 *b) {

  return;

}

#endif

/* select next queue entry based on alias algo - fast! */

inline u32 select_next_queue_entry(afl_state_t *afl) {

  u32    s = rand_below(afl, afl->queued_items);
  double p = rand_next_percent(afl);
  
  /*
  fprintf(stderr, "select: p=%f s=%u ... p < prob[s]=%f ? s=%u : alias[%u]=%u"
  " ==> %u\n", p, s, afl->alias_probability[s], s, s, afl->alias_table[s], p <
  afl->alias_probability[s] ? s : afl->alias_table[s]);
  */
  
  return (p < afl->alias_probability[s] ? s : afl->alias_table[s]);

}

void softmax(double *input, int input_len)
{
  int i;
  double m;
  /* Find maximum value from input array */
  m = input[0];
  for (i = 1; i < input_len; i++) {
    if (input[i] > m) {
      m = input[i];
    }
  }

  double sum = 0;
  for (i = 0; i < input_len; i++) {
    sum += expf(input[i]-m);
  }

  for (i = 0; i < input_len; i++) {
    input[i] = expf(input[i] - m - log(sum));
  }    
}

void softmax_scatter(u8 *input, u32 input_len, afl_state_t *afl)
{
  u32 i;
  double m;
  u32 cnt_table[256] = {0};
  double res_table[256] = {-1.0};
  /* Find maximum value from input array */
  m = input[0];
  cnt_table[input[0]] ++;
  for (i = 1; i < input_len; i++) {
    cnt_table[input[i]] ++;
    if (input[i] > m) {
      m = input[i];
    }
  }

  double sum = 0;
  for(size_t v = 0; v < 256; v++) {
    if (cnt_table[v] == 0) continue;
    sum += expf((float)(v-m)) * cnt_table[v];
  }

  double max_v = expf(m - m - log(sum));
  double min_v = expf(0.0 - m - log(sum));

  for (i = 0; i < input_len; i++) {
    if (res_table[input[i]] == -1) {
      input[i] = (expf((double)input[i] - m - log(sum)) - min_v) / (max_v - min_v) * 255;
      res_table[input[i]] = input[i];
    }
    else {
      input[i] = res_table[input[i]];
    }
  }
}

void construct_connection(afl_state_t *afl) {
  
  while (1){
    if (afl->nn_mem_fd == -1) {
      u64 tmp;
      u64 MAXSIZE = afl->fsrv.map_size;

      while (1) {
        tmp = (u64)random();
        snprintf(afl->nn_mem_name, 1024, "/memfd:%08llx", tmp);
        if (access(afl->nn_mem_name, F_OK) != 0) {
          break;
        }
      }

      int fd = memfd_create(strstr(afl->nn_mem_name, "memfd:") + strlen("memfd:"), MFD_ALLOW_SEALING);
      if (fd == -1) {
        WARNF("memfd_create ERROR!");
        sleep(10);
        continue;
      }
      ftruncate(fd, MAXSIZE);
      snprintf(afl->nn_mem_path, 1024, "/proc/%d/fd/%d", getpid(), fd);
      afl->nn_mem_fd = fd;

      if (ftruncate(fd, MAXSIZE) == -1) {
        WARNF("ftruncate ERROR!");
        close(afl->nn_mem_fd);
        afl->nn_mem_fd = -1;
        sleep(10);
        continue;
      }

      afl->nn_shm = mmap(NULL, MAXSIZE, PROT_WRITE|PROT_READ, MAP_SHARED, fd, 0);
      if (afl->nn_shm == NULL) {
        WARNF("mmap ERROR!");
        close(afl->nn_mem_fd);
        afl->nn_mem_fd = -1;
        sleep(10);
        continue;
      }

    }

    if (afl->nn_unix_sock == -1) {
      struct sockaddr_un un;
      int sock_fd;
      un.sun_family = AF_UNIX;
      char tmp[8] = {0};
      strcpy(un.sun_path,afl->nn_unix_name);

      sock_fd = socket(AF_UNIX,SOCK_STREAM,0);

      if(sock_fd < 0){
        WARNF("request socket ERROR!");
        sleep(10);
        continue;
      }

      if(connect(sock_fd,(struct sockaddr *)&un,sizeof(un)) < 0){
        WARNF("connect socket ERROR!");
        sleep(10);
        continue;
      }

      afl->nn_unix_sock = sock_fd;

      if (send(afl->nn_unix_sock, afl->nn_mem_path, strlen(afl->nn_mem_path), 0) < strlen(afl->nn_mem_path)) {
        WARNF("nn_mem_path send ERROR!");
        close(afl->nn_unix_sock);
        afl->nn_unix_sock = -1;
        sleep(10);
        continue;
      }

      if (read(afl->nn_unix_sock, tmp, 2) < 2) {
        close(afl->nn_unix_sock);
        afl->nn_unix_sock = -1;
        sleep(10);
        continue;
        WARNF("nn_mem_path read ERROR!");
      }
    }
    break;
  }
}

void reconstruct_conn(afl_state_t *afl) {
  u64 MAXSIZE = afl->fsrv.real_map_size;
  if (afl->nn_mem_fd != -1) {
    close(afl->nn_mem_fd);
    afl->nn_mem_fd = -1;
  }
  if (afl->nn_unix_sock != -1) {
    close(afl->nn_unix_sock);
    afl->nn_unix_sock = -1;
  }
  if (afl->nn_shm != NULL) {
    munmap(afl->nn_shm, MAXSIZE);
    afl->nn_shm = NULL;
  }
  memset(afl->nn_mem_path, 0, 1025);
  memset(afl->nn_mem_name, 0, 1025);
  construct_connection(afl);

}

double compute_weight_legacy(afl_state_t *afl, struct queue_entry *q,
                      double avg_exec_us, double avg_bitmap_size,
                      double avg_top_size) {

  double weight = 1.0;

  if (likely(afl->schedule >= FAST && afl->schedule <= RARE)) {

    u32 hits = afl->n_fuzz[q->n_fuzz_entry];
    if (likely(hits)) { weight /= (log10(hits) + 1); }

  }

  if (likely(afl->schedule < RARE)) { weight *= (avg_exec_us / q->exec_us); }
  weight *= (log(q->bitmap_size) / avg_bitmap_size);
  weight *= (1 + (q->tc_ref / avg_top_size));
  if (unlikely(weight < 1.0)) { weight = 1.0; }
  if (unlikely(q->favored)) { weight *= 5; }
  if (unlikely(!q->was_fuzzed)) { weight *= 2; }

  return weight;

}

/* create the alias table that allows weighted random selection - expensive */

double o_distance(u8 *A, u8 *AIM, size_t size) {
  double sum = 0;
  for (size_t i = 0; i < size / 8; i ++) {
    if (*(u64 *)(A+i*8) == *(u64 *)(AIM+i*8)) {
      continue;
    }
    for (size_t j = 0; j < 8; j ++) {
      sum += (A[i*8+j] - AIM[i*8+j]) * (A[i*8+j] - AIM[i*8+j]);
    }
  }
  return sqrt(sum);
}

void create_alias_table(afl_state_t *afl) {

  u32    n = afl->queued_items, i = 0, a, g;
  double sum = 0;

  afl->alias_table =
      (u32 *)afl_realloc((void **)&afl->alias_table, n * sizeof(u32));
  afl->alias_probability = (double *)afl_realloc(
      (void **)&afl->alias_probability, n * sizeof(double));
  double *P = (double *)afl_realloc(AFL_BUF_PARAM(out), n * sizeof(double));
  int    *S = (u32 *)afl_realloc(AFL_BUF_PARAM(out_scratch), n * sizeof(u32));
  int    *L = (u32 *)afl_realloc(AFL_BUF_PARAM(in_scratch), n * sizeof(u32));

  if (!P || !S || !L || !afl->alias_table || !afl->alias_probability) {

    FATAL("could not acquire memory for alias table");

  }

  memset((void *)afl->alias_table, 0, n * sizeof(u32));
  memset((void *)afl->alias_probability, 0, n * sizeof(double));

  {
    double avg_exec_us = 0.0;
    double avg_bitmap_size = 0.0;
    double avg_top_size = 0.0;
    u32    active = 0;

    for (i = 0; i < n; i++) {

      struct queue_entry *q = afl->queue_buf[i];

      // disabled entries might have timings and bitmap values
      if (likely(!q->disabled)) {

        avg_exec_us += q->exec_us;
        avg_bitmap_size += log(q->bitmap_size);
        avg_top_size += q->tc_ref;
        ++active;

      }

    }

    avg_exec_us /= active;
    avg_bitmap_size /= active;
    avg_top_size /= active;

    while (1) {
      //send virgin_bits
      int re_do = 0;
      char recv_buf[0x10] = {0};
      for (size_t idx = 0; idx < afl->queued_items; idx ++) {
        if (time(NULL) < afl->queue_buf[idx]->weight_update_time + 15 * 60 &&
            afl->queue_buf[idx]->AI_weight != -99999999.0) {
          //printf("skip %lu: %lu -> %lu, now: %lf\n", 
          //  idx, time(NULL), afl->queue_buf[idx]->weight_update_time, afl->queue_buf[idx]->AI_weight
          //);
          continue;
        }
        if (send(afl->nn_unix_sock, "get weit", strlen("get weit"), 0) < strlen("get weit")) {
          WARNF("get weit send ERROR!");
          reconstruct_conn(afl);
          continue;
        }
        if (read(afl->nn_unix_sock, recv_buf, 2) < 2) {
          WARNF("get weit read ERROR!");
          reconstruct_conn(afl);
          continue;
        }
        *(size_t *)recv_buf = idx;
        if (send(afl->nn_unix_sock, recv_buf, 8, 0) < 8) {
          WARNF("idx send ERROR!");
          reconstruct_conn(afl);
          continue;
        }
        if (read(afl->nn_unix_sock, recv_buf, 8) < 8) {
          WARNF("weight read ERROR!");
          reconstruct_conn(afl);
          continue;
        }
        afl->queue_buf[idx]->AI_weight = (*(double *)recv_buf);
        if (afl->queue_buf[idx]->AI_weight == -99999999.0) {
          if (send(afl->nn_unix_sock, "LEAVE", 5, 0) < 5) {
            WARNF("LEAVE send ERROR!");
            reconstruct_conn(afl);
            continue;
          }
          continue;
        }
        if (send(afl->nn_unix_sock, "ATTEN", 5, 0) < 5) {
          WARNF("ATTEN send ERROR!");
          reconstruct_conn(afl);
          continue;
        }
        if (read(afl->nn_unix_sock, recv_buf, 8) < 8) {
          WARNF("ATTEN read ERROR!");
          reconstruct_conn(afl);
          continue;
        }
        size_t attention_size = *(size_t *)recv_buf;
        if (afl->queue_buf[idx]->attention_nodes == NULL) {
          afl->queue_buf[idx]->attention_nodes = (u8 *)calloc(afl->fsrv.map_size, sizeof(u8));
        }
        u8 *attention_tmp = calloc(attention_size, 1);
        memcpy(attention_tmp, afl->nn_shm, attention_size);
        size_t target_size = afl->fsrv.map_size;
        int res = uncompress(afl->queue_buf[idx]->attention_nodes, &target_size, attention_tmp, attention_size);
        if (res != Z_OK || target_size != afl->fsrv.map_size) {
          WARNF("ZLIB DECOMPRESS ERROR %d %lu, %lu", res, idx, target_size);
        }
        free(attention_tmp);
        attention_tmp = NULL;
        //softmax_scatter(afl->queue_buf[idx]->attention_nodes, afl->fsrv.map_size, afl);
        for (size_t ii = 0; ii < afl->fsrv.map_size; ii ++) {
          afl->queue_buf[idx]->attention_nodes[ii] = afl->queue_buf[idx]->attention_nodes[ii] * 0xff / 0x10;
        }
        afl->queue_buf[idx]->weight_update_time = time(NULL);
        //afl->queue_buf[idx]->attention_update_time = time(NULL);
      }
      afl->virgin_cnt = count_non_255_bytes(afl, afl->virgin_bits);

      sum = 0;
      double max_ai_weight = -99999999.0, min_ai_weight = 9999999;

      for (size_t seed_idx = 0; seed_idx < afl->queued_items; seed_idx ++) {
        //if (afl->queue_buf[seed_idx]->AI_weight < 100) afl->queue_buf[seed_idx]->AI_weight = 100;
        if (afl->queue_buf[seed_idx]->AI_weight != -99999999.0 && afl->queue_buf[seed_idx]->AI_weight > max_ai_weight) max_ai_weight = afl->queue_buf[seed_idx]->AI_weight;
        if (afl->queue_buf[seed_idx]->AI_weight != -99999999.0 && afl->queue_buf[seed_idx]->AI_weight < min_ai_weight) min_ai_weight = afl->queue_buf[seed_idx]->AI_weight;
      }

      if (min_ai_weight >= max_ai_weight) {
        max_ai_weight = min_ai_weight + 1;
      }

      WARNF("max:%lf, min:%lf", max_ai_weight, min_ai_weight);
      if ((get_cur_time() > afl->average_attention_update_time + 15 * 60 * 1000) || 
          afl->average_attention == NULL)
      {
        printf("update average_attention_update_time, prev: %llu\n", afl->average_attention_update_time);
        afl->average_attention_update_time = get_cur_time();
        if (afl->average_attention == NULL) {
          afl->average_attention = calloc(afl->fsrv.map_size, sizeof(double));
        }
        else {
          memset(afl->average_attention, 0, sizeof(double) * afl->fsrv.map_size);
        }
        //double max_a = 0, min_a = 1000000;
        size_t seed_att_cnt = 0;
        for (size_t seed_idx = 0; seed_idx < afl->queued_items; seed_idx ++) {
          if (!afl->queue_buf[seed_idx]->attention_nodes) {
            afl->queue_buf[seed_idx]->attention_score = -1;
            continue;
          }
          seed_att_cnt ++;
          for (size_t hash_idx = 0; hash_idx < afl->fsrv.map_size; hash_idx ++) {
            afl->average_attention[hash_idx] += afl->queue_buf[seed_idx]->attention_nodes[hash_idx];
            //if (afl->average_attention[hash_idx] < min_a) {
            //  min_a = afl->average_attention[hash_idx];
            //}
            //if (afl->average_attention[hash_idx] > max_a) {
            //  max_a = afl->average_attention[hash_idx];
            //}
          }
        }
        //if (min_a >= max_a) {
        //  max_a = min_a + 1;
        //}
        if (afl->average_attention_scatter == NULL) {
          afl->average_attention_scatter = calloc(afl->fsrv.map_size, sizeof(u8));
        }
        else {
          memset(afl->average_attention_scatter, 0, sizeof(u8) * afl->fsrv.map_size);
        }
        for (size_t hash_idx = 0; hash_idx < afl->fsrv.map_size; hash_idx ++) {
          afl->average_attention_scatter[hash_idx] = 
            afl->average_attention[hash_idx] * 0xff / 0x10 / seed_att_cnt;
        }
        for (size_t ii = 0; ii < 100; ii ++) {
          printf("%02x", afl->average_attention_scatter[ii]);
        }
        printf("\n");
        //softmax_scatter(afl->average_attention_scatter, afl->fsrv.map_size, afl);
      }

      double max_dist = 0;
      double min_dist = 9999999999999;
      for (size_t seed_idx = 0; seed_idx < afl->queued_items; seed_idx ++) {
        if (!afl->queue_buf[seed_idx]->attention_nodes) {
          afl->queue_buf[seed_idx]->attention_score = -1;
          continue;
        }
        //printf("queue attention_update_time %lu, afl average_attention_update_time %lu\n", 
        //  afl->queue_buf[seed_idx]->attention_update_time, afl->average_attention_update_time
        //);
        if (afl->queue_buf[seed_idx]->attention_update_time != afl->average_attention_update_time) {
          afl->queue_buf[seed_idx]->attention_score = o_distance(
            afl->queue_buf[seed_idx]->attention_nodes, 
            afl->average_attention_scatter, 
            afl->fsrv.map_size
          );
          afl->queue_buf[seed_idx]->attention_update_time = afl->average_attention_update_time;
          printf("update attention_score for seed %lu using %lf\n", seed_idx, afl->queue_buf[seed_idx]->attention_score);
        }
        if (afl->queue_buf[seed_idx]->attention_score > max_dist) {
          max_dist = afl->queue_buf[seed_idx]->attention_score;
        }
        if (afl->queue_buf[seed_idx]->attention_score < min_dist) {
          min_dist = afl->queue_buf[seed_idx]->attention_score;
        }
      }
      if (max_dist == 0) {
        max_dist = 1;
      }
      if (min_dist >= max_dist) {
        max_dist = min_dist + 1;
      }
      //WARNF("max:%lf, min:%lf", max_dist, min_dist);
      for (size_t seed_idx = 0; seed_idx < afl->queued_items; seed_idx ++) {
        if (!afl->queue_buf[seed_idx]->attention_nodes || afl->queue_buf[seed_idx]->attention_score <= 0) {
          afl->queue_buf[seed_idx]->attention_score_norm = 0.5;
          continue;
        }
        afl->queue_buf[seed_idx]->attention_score_norm = (afl->queue_buf[seed_idx]->attention_score - min_dist) / (max_dist - min_dist);
        if (afl->queue_buf[seed_idx]->attention_score_norm == 0) {
          afl->queue_buf[seed_idx]->attention_score_norm = 0.5;
        }
      }

      for (size_t seed_idx = 0; seed_idx < afl->queued_items; seed_idx ++) {
        //afl->queue_buf[seed_idx]->legacy_weight = compute_weight_legacy(afl, afl->queue_buf[seed_idx], avg_exec_us, avg_bitmap_size, avg_top_size);
        if (afl->queue_buf[seed_idx]->AI_weight == -99999999.0) {
          afl->queue_buf[seed_idx]->AI_weight_norm = 0.5;
        }
        else {
          afl->queue_buf[seed_idx]->AI_weight_norm = 
            (afl->queue_buf[seed_idx]->AI_weight - min_ai_weight) / (max_ai_weight - min_ai_weight);
        }
        afl->queue_buf[seed_idx]->weight = 10000000.0 * 
        (
          pow(afl->queue_buf[seed_idx]->AI_weight_norm, 2 - afl->queue_buf[seed_idx]->attention_score_norm)
        ) /
        (afl->queue_buf[seed_idx]->exec_us + 10);
        sum += afl->queue_buf[seed_idx]->weight;
      }


      if (re_do) {
        continue;
      }

      //calculate perf_score
      for (i = 0; i < n; i++) {

        struct queue_entry *q = afl->queue_buf[i];

        if (likely(!q->disabled)) {

          q->perf_score = calculate_score(afl, q);

        }

      }

      for (i = 0; i < n; i++) {

        // weight is always 0 for disabled entries
        P[i] = (afl->queue_buf[i]->weight * n) / sum;

      }
      break;
    }
    
  }
  

  int nS = 0, nL = 0, s;
  for (s = (s32)n - 1; s >= 0; --s) {

    if (P[s] < 1) {

      S[nS++] = s;

    } else {

      L[nL++] = s;

    }

  }

  while (nS && nL) {

    a = S[--nS];
    g = L[--nL];
    afl->alias_probability[a] = P[a];
    afl->alias_table[a] = g;
    P[g] = P[g] + P[a] - 1;
    if (P[g] < 1) {

      S[nS++] = g;

    } else {

      L[nL++] = g;

    }

  }

  while (nL)
    afl->alias_probability[L[--nL]] = 1;

  while (nS)
    afl->alias_probability[S[--nS]] = 1;

  afl->reinit_table = 0;

  /*
  #ifdef INTROSPECTION
    u8 fn[PATH_MAX];
    snprintf(fn, PATH_MAX, "%s/introspection_corpus.txt", afl->out_dir);
    FILE *f = fopen(fn, "a");
    if (f) {

      for (i = 0; i < n; i++) {

        struct queue_entry *q = afl->queue_buf[i];
        fprintf(
            f,
            "entry=%u name=%s favored=%s variable=%s disabled=%s len=%u "
            "exec_us=%u "
            "bitmap_size=%u bitsmap_size=%u tops=%u weight=%f perf_score=%f\n",
            i, q->fname, q->favored ? "true" : "false",
            q->var_behavior ? "true" : "false", q->disabled ? "true" : "false",
            q->len, (u32)q->exec_us, q->bitmap_size, q->bitsmap_size, q->tc_ref,
            q->weight, q->perf_score);

      }

      fprintf(f, "\n");
      fclose(f);

    }

  #endif
  */
  /*
  fprintf(stderr, "  entry  alias  probability  perf_score   weight
  filename\n"); for (u32 i = 0; i < n; ++i) fprintf(stderr, "  %5u  %5u  %11u
  %0.9f  %0.9f  %s\n", i, afl->alias_table[i], afl->alias_probability[i],
  afl->queue_buf[i]->perf_score, afl->queue_buf[i]->weight,
            afl->queue_buf[i]->fname);
  */

}


/* Mark deterministic checks as done for a particular queue entry. We use the
   .state file to avoid repeating deterministic fuzzing when resuming aborted
   scans. */

void mark_as_det_done(afl_state_t *afl, struct queue_entry *q) {

  char fn[PATH_MAX];
  s32  fd;

  snprintf(fn, PATH_MAX, "%s/queue/.state/deterministic_done/%s", afl->out_dir,
           strrchr((char *)q->fname, '/') + 1);

  fd = open(fn, O_WRONLY | O_CREAT | O_EXCL, DEFAULT_PERMISSION);
  if (fd < 0) { PFATAL("Unable to create '%s'", fn); }
  close(fd);

  q->passed_det = 1;

}

/* Mark as variable. Create symlinks if possible to make it easier to examine
   the files. */

void mark_as_variable(afl_state_t *afl, struct queue_entry *q) {

  char fn[PATH_MAX];
  char ldest[PATH_MAX];

  char *fn_name = strrchr((char *)q->fname, '/') + 1;

  sprintf(ldest, "../../%s", fn_name);
  sprintf(fn, "%s/queue/.state/variable_behavior/%s", afl->out_dir, fn_name);

  if (symlink(ldest, fn)) {

    s32 fd = open(fn, O_WRONLY | O_CREAT | O_EXCL, DEFAULT_PERMISSION);
    if (fd < 0) { PFATAL("Unable to create '%s'", fn); }
    close(fd);

  }

  q->var_behavior = 1;

}

/* Mark / unmark as redundant (edge-only). This is not used for restoring state,
   but may be useful for post-processing datasets. */

void mark_as_redundant(afl_state_t *afl, struct queue_entry *q, u8 state) {

  if (likely(state == q->fs_redundant)) { return; }

  char fn[PATH_MAX];

  q->fs_redundant = state;

  sprintf(fn, "%s/queue/.state/redundant_edges/%s", afl->out_dir,
          strrchr((char *)q->fname, '/') + 1);

  if (state) {

    s32 fd;

    fd = open(fn, O_WRONLY | O_CREAT | O_EXCL, DEFAULT_PERMISSION);
    if (fd < 0) { PFATAL("Unable to create '%s'", fn); }
    close(fd);

  } else {

    if (unlink(fn)) { PFATAL("Unable to remove '%s'", fn); }

  }

}

/* check if pointer is ascii or UTF-8 */

u8 check_if_text_buf(u8 *buf, u32 len) {

  u32 offset = 0, ascii = 0, utf8 = 0;

  while (offset < len) {

    // ASCII: <= 0x7F to allow ASCII control characters
    if ((buf[offset + 0] == 0x09 || buf[offset + 0] == 0x0A ||
         buf[offset + 0] == 0x0D ||
         (0x20 <= buf[offset + 0] && buf[offset + 0] <= 0x7E))) {

      offset++;
      utf8++;
      ascii++;
      continue;

    }

    if (isascii((int)buf[offset]) || isprint((int)buf[offset])) {

      ascii++;
      // we continue though as it can also be a valid utf8

    }

    // non-overlong 2-byte
    if (len - offset > 1 &&
        ((0xC2 <= buf[offset + 0] && buf[offset + 0] <= 0xDF) &&
         (0x80 <= buf[offset + 1] && buf[offset + 1] <= 0xBF))) {

      offset += 2;
      utf8++;
      continue;

    }

    // excluding overlongs
    if ((len - offset > 2) &&
        ((buf[offset + 0] == 0xE0 &&
          (0xA0 <= buf[offset + 1] && buf[offset + 1] <= 0xBF) &&
          (0x80 <= buf[offset + 2] &&
           buf[offset + 2] <= 0xBF)) ||  // straight 3-byte
         (((0xE1 <= buf[offset + 0] && buf[offset + 0] <= 0xEC) ||
           buf[offset + 0] == 0xEE || buf[offset + 0] == 0xEF) &&
          (0x80 <= buf[offset + 1] && buf[offset + 1] <= 0xBF) &&
          (0x80 <= buf[offset + 2] &&
           buf[offset + 2] <= 0xBF)) ||  // excluding surrogates
         (buf[offset + 0] == 0xED &&
          (0x80 <= buf[offset + 1] && buf[offset + 1] <= 0x9F) &&
          (0x80 <= buf[offset + 2] && buf[offset + 2] <= 0xBF)))) {

      offset += 3;
      utf8++;
      continue;

    }

    // planes 1-3
    if ((len - offset > 3) &&
        ((buf[offset + 0] == 0xF0 &&
          (0x90 <= buf[offset + 1] && buf[offset + 1] <= 0xBF) &&
          (0x80 <= buf[offset + 2] && buf[offset + 2] <= 0xBF) &&
          (0x80 <= buf[offset + 3] &&
           buf[offset + 3] <= 0xBF)) ||  // planes 4-15
         ((0xF1 <= buf[offset + 0] && buf[offset + 0] <= 0xF3) &&
          (0x80 <= buf[offset + 1] && buf[offset + 1] <= 0xBF) &&
          (0x80 <= buf[offset + 2] && buf[offset + 2] <= 0xBF) &&
          (0x80 <= buf[offset + 3] && buf[offset + 3] <= 0xBF)) ||  // plane 16
         (buf[offset + 0] == 0xF4 &&
          (0x80 <= buf[offset + 1] && buf[offset + 1] <= 0x8F) &&
          (0x80 <= buf[offset + 2] && buf[offset + 2] <= 0xBF) &&
          (0x80 <= buf[offset + 3] && buf[offset + 3] <= 0xBF)))) {

      offset += 4;
      utf8++;
      continue;

    }

    offset++;

  }

  return (utf8 > ascii ? utf8 : ascii);

}

/* check if queue entry is ascii or UTF-8 */

static u8 check_if_text(afl_state_t *afl, struct queue_entry *q) {

  if (q->len < AFL_TXT_MIN_LEN || q->len < AFL_TXT_MAX_LEN) return 0;

  u8     *buf;
  int     fd;
  u32     len = q->len, offset = 0, ascii = 0, utf8 = 0;
  ssize_t comp;

  if (len >= MAX_FILE) len = MAX_FILE - 1;
  if ((fd = open((char *)q->fname, O_RDONLY)) < 0) return 0;
  buf = (u8 *)afl_realloc(AFL_BUF_PARAM(in_scratch), len + 1);
  comp = read(fd, buf, len);
  close(fd);
  if (comp != (ssize_t)len) return 0;
  buf[len] = 0;

  while (offset < len) {

    // ASCII: <= 0x7F to allow ASCII control characters
    if ((buf[offset + 0] == 0x09 || buf[offset + 0] == 0x0A ||
         buf[offset + 0] == 0x0D ||
         (0x20 <= buf[offset + 0] && buf[offset + 0] <= 0x7E))) {

      offset++;
      utf8++;
      ascii++;
      continue;

    }

    if (isascii((int)buf[offset]) || isprint((int)buf[offset])) {

      ascii++;
      // we continue though as it can also be a valid utf8

    }

    // non-overlong 2-byte
    if (len - offset > 1 &&
        ((0xC2 <= buf[offset + 0] && buf[offset + 0] <= 0xDF) &&
         (0x80 <= buf[offset + 1] && buf[offset + 1] <= 0xBF))) {

      offset += 2;
      utf8++;
      comp--;
      continue;

    }

    // excluding overlongs
    if ((len - offset > 2) &&
        ((buf[offset + 0] == 0xE0 &&
          (0xA0 <= buf[offset + 1] && buf[offset + 1] <= 0xBF) &&
          (0x80 <= buf[offset + 2] &&
           buf[offset + 2] <= 0xBF)) ||  // straight 3-byte
         (((0xE1 <= buf[offset + 0] && buf[offset + 0] <= 0xEC) ||
           buf[offset + 0] == 0xEE || buf[offset + 0] == 0xEF) &&
          (0x80 <= buf[offset + 1] && buf[offset + 1] <= 0xBF) &&
          (0x80 <= buf[offset + 2] &&
           buf[offset + 2] <= 0xBF)) ||  // excluding surrogates
         (buf[offset + 0] == 0xED &&
          (0x80 <= buf[offset + 1] && buf[offset + 1] <= 0x9F) &&
          (0x80 <= buf[offset + 2] && buf[offset + 2] <= 0xBF)))) {

      offset += 3;
      utf8++;
      comp -= 2;
      continue;

    }

    // planes 1-3
    if ((len - offset > 3) &&
        ((buf[offset + 0] == 0xF0 &&
          (0x90 <= buf[offset + 1] && buf[offset + 1] <= 0xBF) &&
          (0x80 <= buf[offset + 2] && buf[offset + 2] <= 0xBF) &&
          (0x80 <= buf[offset + 3] &&
           buf[offset + 3] <= 0xBF)) ||  // planes 4-15
         ((0xF1 <= buf[offset + 0] && buf[offset + 0] <= 0xF3) &&
          (0x80 <= buf[offset + 1] && buf[offset + 1] <= 0xBF) &&
          (0x80 <= buf[offset + 2] && buf[offset + 2] <= 0xBF) &&
          (0x80 <= buf[offset + 3] && buf[offset + 3] <= 0xBF)) ||  // plane 16
         (buf[offset + 0] == 0xF4 &&
          (0x80 <= buf[offset + 1] && buf[offset + 1] <= 0x8F) &&
          (0x80 <= buf[offset + 2] && buf[offset + 2] <= 0xBF) &&
          (0x80 <= buf[offset + 3] && buf[offset + 3] <= 0xBF)))) {

      offset += 4;
      utf8++;
      comp -= 3;
      continue;

    }

    offset++;

  }

  u32 percent_utf8 = (utf8 * 100) / comp;
  u32 percent_ascii = (ascii * 100) / len;

  if (percent_utf8 >= percent_ascii && percent_utf8 >= AFL_TXT_MIN_PERCENT)
    return 2;
  if (percent_ascii >= AFL_TXT_MIN_PERCENT) return 1;
  return 0;

}

/* Append new test case to the queue. */

void add_to_queue(afl_state_t *afl, u8 *fname, u32 len, u8 passed_det) {

  struct queue_entry *q =
      (struct queue_entry *)ck_alloc(sizeof(struct queue_entry));

  q->fname = fname;
  q->len = len;
  q->depth = afl->cur_depth + 1;
  q->passed_det = passed_det;
  q->trace_mini = NULL;
  q->testcase_buf = NULL;
  q->mother = afl->queue_cur;

#ifdef INTROSPECTION
  q->bitsmap_size = afl->bitsmap_size;
#endif

  if (q->depth > afl->max_depth) { afl->max_depth = q->depth; }

  if (afl->queue_top) {

    afl->queue_top = q;

  } else {

    afl->queue = afl->queue_top = q;

  }

  if (likely(q->len > 4)) afl->ready_for_splicing_count++;

  ++afl->queued_items;
  ++afl->active_items;
  ++afl->pending_not_fuzzed;

  afl->cycles_wo_finds = 0;

  struct queue_entry **queue_buf = (struct queue_entry **)afl_realloc(
      AFL_BUF_PARAM(queue), afl->queued_items * sizeof(struct queue_entry *));
  if (unlikely(!queue_buf)) { PFATAL("alloc"); }
  queue_buf[afl->queued_items - 1] = q;
  q->id = afl->queued_items - 1;

  afl->last_find_time = get_cur_time();

  if (afl->custom_mutators_count) {

    /* At the initialization stage, queue_cur is NULL */
    if (afl->queue_cur && !afl->syncing_party) {

      run_afl_custom_queue_new_entry(afl, q, fname, afl->queue_cur->fname);

    }

  }

  /* only redqueen currently uses is_ascii */
  if (unlikely(afl->shm.cmplog_mode && !q->is_ascii)) {

    q->is_ascii = check_if_text(afl, q);

  }

  /*async calculate queue weight*/
  size_t start_time = get_cur_time();
  if (1) {
    
    struct queue_entry *q = afl->queue_buf[afl->queued_items - 1];

    if (q->trace_mini_kept == NULL) {    
      if (q->mother != NULL) {
        //calibrate_case(afl, q, mem, afl->queue_cycle - 1, 0);
        q->trace_mini_kept = (u8 *)calloc(afl->fsrv.map_size >> 3, 1);
        minimize_bits(afl, q->trace_mini_kept, afl->fsrv.trace_bits);
      }
      else {
        q->trace_mini_kept = (u8 *)calloc(afl->fsrv.map_size >> 3, 1);
      }
      q->trace_mini_kept_len = afl->fsrv.map_size >> 3;
      //zlib compress
      u8 *compressed = (u8 *)calloc(afl->fsrv.map_size >> 3, 1);
      size_t compressed_len = afl->fsrv.map_size >> 3;
      int comp_res = compress(compressed,&compressed_len,q->trace_mini_kept,q->trace_mini_kept_len);
      if (comp_res != Z_OK) {
        WARNF("zlib compress error %d", comp_res);
      }
      free(q->trace_mini_kept);
      q->trace_mini_kept = compressed;
      q->trace_mini_kept_len = compressed_len;
    }

    while (1) {
      char recv_buf[0x10] = {0};
      if (
          afl->virgin_upload_time == 0 || 
          afl->virgin_upload_time > get_cur_time() ||
          afl->virgin_upload_time + 15 * 60 < get_cur_time() ||
          afl->virgin_changed
      ) {
        memcpy(afl->nn_shm, afl->virgin_bits, afl->fsrv.map_size);
        if (send(afl->nn_unix_sock, "virg now", strlen("virg now"), 0) < strlen("virg now")) {
          WARNF("virgin_bits send ERROR!");
          reconstruct_conn(afl);
          continue;
        }
        if (read(afl->nn_unix_sock, recv_buf, 2) < 2) {
          WARNF("virgin_bits read ERROR!");
          reconstruct_conn(afl);
          continue;
        }
        afl->virgin_changed = 0;
      }

      if (send(afl->nn_unix_sock, "chck idx", strlen("chck idx"), 0) < strlen("chck idx")) {
        WARNF("check idx send ERROR!");
        reconstruct_conn(afl);
        continue;
      }
      if (read(afl->nn_unix_sock, recv_buf, 8) < 8) {
        WARNF("check idx read ERROR!");
        reconstruct_conn(afl);
        continue;
      }
      size_t idx_got = *(size_t *)recv_buf;
      //WARNF("%lu, %lu\n", idx_got, afl->queued_items - 1);
      
      if (idx_got != afl->queued_items - 1) {
        if (send(afl->nn_unix_sock, "clr seed", strlen("clr seed"), 0) < strlen("clr seed")) {
          WARNF("clr seed send ERROR!");
          reconstruct_conn(afl);
          continue;
        }
        if (read(afl->nn_unix_sock, recv_buf, 2) < 2) {
          WARNF("clr seed read ERROR!");
          reconstruct_conn(afl);
          continue;
        }
        for(size_t idx = 0; idx < afl->queued_items; idx ++) {
          memcpy(afl->nn_shm, afl->queue_buf[idx]->trace_mini_kept, afl->queue_buf[idx]->trace_mini_kept_len);
          if (send(afl->nn_unix_sock, "add seed", strlen("add seed"), 0) < strlen("add seed")) {
            WARNF("add seed ERROR!");
            reconstruct_conn(afl);
            continue;
          }
          if (read(afl->nn_unix_sock, recv_buf, 2) < 2) {
            WARNF("add seed read ERROR!");
            reconstruct_conn(afl);
            continue;
          }
          *(size_t *)recv_buf = afl->queue_buf[idx]->trace_mini_kept_len;
          if (send(afl->nn_unix_sock, recv_buf, 8, 0) < 8) {
            WARNF("add seed size send ERROR!");
            reconstruct_conn(afl);
            continue;
          }
          if (read(afl->nn_unix_sock, recv_buf, 2) < 2) {
            WARNF("add seed ok ERROR!");
            reconstruct_conn(afl);
            continue;
          }
        }
      }
      else {
        size_t idx = afl->queued_items - 1;
          memcpy(afl->nn_shm, afl->queue_buf[idx]->trace_mini_kept, afl->queue_buf[idx]->trace_mini_kept_len);
          if (send(afl->nn_unix_sock, "add seed", strlen("add seed"), 0) < strlen("add seed")) {
            WARNF("add seed ERROR!");
            reconstruct_conn(afl);
            continue;
          }
          if (read(afl->nn_unix_sock, recv_buf, 2) < 2) {
            WARNF("add seed read ERROR!");
            reconstruct_conn(afl);
            continue;
          }
          *(size_t *)recv_buf = afl->queue_buf[idx]->trace_mini_kept_len;
          if (send(afl->nn_unix_sock, recv_buf, 8, 0) < 8) {
            WARNF("add seed size send ERROR!");
            reconstruct_conn(afl);
            continue;
          }
          if (read(afl->nn_unix_sock, recv_buf, 2) < 2) {
            WARNF("add seed ok ERROR!");
            reconstruct_conn(afl);
            continue;
          }
      }
      break;
    }
  }
  afl->overhead += (get_cur_time() - start_time) / 1000.0;
  //WARNF("queue add use %lf sec", (get_cur_time() - start_time)/1000.0);

}

/* Destroy the entire queue. */

void destroy_queue(afl_state_t *afl) {

  u32 i;

  for (i = 0; i < afl->queued_items; i++) {

    struct queue_entry *q;

    q = afl->queue_buf[i];
    ck_free(q->fname);
    ck_free(q->trace_mini);
    ck_free(q->trace_mini_kept);
    ck_free(q);

  }

}

/* When we bump into a new path, we call this to see if the path appears
   more "favorable" than any of the existing ones. The purpose of the
   "favorables" is to have a minimal set of paths that trigger all the bits
   seen in the bitmap so far, and focus on fuzzing them at the expense of
   the rest.

   The first step of the process is to maintain a list of afl->top_rated[]
   entries for every byte in the bitmap. We win that slot if there is no
   previous contender, or if the contender has a more favorable speed x size
   factor. */

void update_bitmap_score(afl_state_t *afl, struct queue_entry *q) {

  u32 i;
  u64 fav_factor;
  u64 fuzz_p2;

  if (unlikely(afl->schedule >= FAST && afl->schedule < RARE))
    fuzz_p2 = 0;  // Skip the fuzz_p2 comparison
  else if (unlikely(afl->schedule == RARE))
    fuzz_p2 = next_pow2(afl->n_fuzz[q->n_fuzz_entry]);
  else
    fuzz_p2 = q->fuzz_level;

  if (unlikely(afl->schedule >= RARE) || unlikely(afl->fixed_seed)) {

    fav_factor = q->len << 2;

  } else {

    fav_factor = q->exec_us * q->len;

  }

  /* For every byte set in afl->fsrv.trace_bits[], see if there is a previous
     winner, and how it compares to us. */
  for (i = 0; i < afl->fsrv.map_size; ++i) {

    if (afl->fsrv.trace_bits[i]) {

      if (afl->top_rated[i]) {

        /* Faster-executing or smaller test cases are favored. */
        u64 top_rated_fav_factor;
        u64 top_rated_fuzz_p2;
        if (unlikely(afl->schedule >= FAST && afl->schedule <= RARE))
          top_rated_fuzz_p2 =
              next_pow2(afl->n_fuzz[afl->top_rated[i]->n_fuzz_entry]);
        else
          top_rated_fuzz_p2 = afl->top_rated[i]->fuzz_level;

        if (unlikely(afl->schedule >= RARE) || unlikely(afl->fixed_seed)) {

          top_rated_fav_factor = afl->top_rated[i]->len << 2;

        } else {

          top_rated_fav_factor =
              afl->top_rated[i]->exec_us * afl->top_rated[i]->len;

        }

        if (fuzz_p2 > top_rated_fuzz_p2) {

          continue;

        } else if (fuzz_p2 == top_rated_fuzz_p2) {

          if (fav_factor > top_rated_fav_factor) { continue; }

        }

        if (unlikely(afl->schedule >= RARE) || unlikely(afl->fixed_seed)) {

          if (fav_factor > afl->top_rated[i]->len << 2) { continue; }

        } else {

          if (fav_factor >
              afl->top_rated[i]->exec_us * afl->top_rated[i]->len) {

            continue;

          }

        }

        /* Looks like we're going to win. Decrease ref count for the
           previous winner, discard its afl->fsrv.trace_bits[] if necessary. */

        if (!--afl->top_rated[i]->tc_ref) {

          ck_free(afl->top_rated[i]->trace_mini);
          afl->top_rated[i]->trace_mini = 0;

        }

      }

      /* Insert ourselves as the new winner. */

      afl->top_rated[i] = q;
      ++q->tc_ref;

      if (!q->trace_mini) {

        u32 len = (afl->fsrv.map_size >> 3);
        q->trace_mini = (u8 *)ck_alloc(len);
        minimize_bits(afl, q->trace_mini, afl->fsrv.trace_bits);

      }

      afl->score_changed = 1;

    }

  }

}

/* The second part of the mechanism discussed above is a routine that
   goes over afl->top_rated[] entries, and then sequentially grabs winners for
   previously-unseen bytes (temp_v) and marks them as favored, at least
   until the next run. The favored entries are given more air time during
   all fuzzing steps. */

void cull_queue(afl_state_t *afl) {

  if (likely(!afl->score_changed || afl->non_instrumented_mode)) { return; }

  u32 len = (afl->fsrv.map_size >> 3);
  u32 i;
  u8 *temp_v = afl->map_tmp_buf;

  afl->score_changed = 0;

  memset(temp_v, 255, len);

  afl->queued_favored = 0;
  afl->pending_favored = 0;

  for (i = 0; i < afl->queued_items; i++) {

    afl->queue_buf[i]->favored = 0;

  }

  /* Let's see if anything in the bitmap isn't captured in temp_v.
     If yes, and if it has a afl->top_rated[] contender, let's use it. */

  for (i = 0; i < afl->fsrv.map_size; ++i) {

    if (afl->top_rated[i] && (temp_v[i >> 3] & (1 << (i & 7)))) {

      u32 j = len;

      /* Remove all bits belonging to the current entry from temp_v. */

      while (j--) {

        if (afl->top_rated[i]->trace_mini[j]) {

          temp_v[j] &= ~afl->top_rated[i]->trace_mini[j];

        }

      }

      if (!afl->top_rated[i]->favored) {

        afl->top_rated[i]->favored = 1;
        ++afl->queued_favored;

        if (!afl->top_rated[i]->was_fuzzed) { ++afl->pending_favored; }

      }

    }

  }

  for (i = 0; i < afl->queued_items; i++) {

    if (likely(!afl->queue_buf[i]->disabled)) {

      mark_as_redundant(afl, afl->queue_buf[i], !afl->queue_buf[i]->favored);

    }

  }

}

/* Calculate case desirability score to adjust the length of havoc fuzzing.
   A helper function for fuzz_one(). Maybe some of these constants should
   go into config.h. */

u32 calculate_score(afl_state_t *afl, struct queue_entry *q) {

  u32 cal_cycles = afl->total_cal_cycles;
  u32 bitmap_entries = afl->total_bitmap_entries;

  if (unlikely(!cal_cycles)) { cal_cycles = 1; }
  if (unlikely(!bitmap_entries)) { bitmap_entries = 1; }

  u32 avg_exec_us = afl->total_cal_us / cal_cycles;
  u32 avg_bitmap_size = afl->total_bitmap_size / bitmap_entries;
  u32 perf_score = 100;

  /* Adjust score based on execution speed of this path, compared to the
     global average. Multiplier ranges from 0.1x to 3x. Fast inputs are
     less expensive to fuzz, so we're giving them more air time. */

  // TODO BUG FIXME: is this really a good idea?
  // This sounds like looking for lost keys under a street light just because
  // the light is better there.
  // Longer execution time means longer work on the input, the deeper in
  // coverage, the better the fuzzing, right? -mh

  if (likely(afl->schedule < RARE) && likely(!afl->fixed_seed)) {

    if (q->exec_us * 0.1 > avg_exec_us) {

      perf_score = 10;

    } else if (q->exec_us * 0.25 > avg_exec_us) {

      perf_score = 25;

    } else if (q->exec_us * 0.5 > avg_exec_us) {

      perf_score = 50;

    } else if (q->exec_us * 0.75 > avg_exec_us) {

      perf_score = 75;

    } else if (q->exec_us * 4 < avg_exec_us) {

      perf_score = 300;

    } else if (q->exec_us * 3 < avg_exec_us) {

      perf_score = 200;

    } else if (q->exec_us * 2 < avg_exec_us) {

      perf_score = 150;

    }

  }

  /* Adjust score based on bitmap size. The working theory is that better
     coverage translates to better targets. Multiplier from 0.25x to 3x. */

  if (q->bitmap_size * 0.3 > avg_bitmap_size) {

    perf_score *= 3;

  } else if (q->bitmap_size * 0.5 > avg_bitmap_size) {

    perf_score *= 2;

  } else if (q->bitmap_size * 0.75 > avg_bitmap_size) {

    perf_score *= 1.5;

  } else if (q->bitmap_size * 3 < avg_bitmap_size) {

    perf_score *= 0.25;

  } else if (q->bitmap_size * 2 < avg_bitmap_size) {

    perf_score *= 0.5;

  } else if (q->bitmap_size * 1.5 < avg_bitmap_size) {

    perf_score *= 0.75;

  }

  /* Adjust score based on handicap. Handicap is proportional to how late
     in the game we learned about this path. Latecomers are allowed to run
     for a bit longer until they catch up with the rest. */

  if (q->handicap >= 4) {

    perf_score *= 4;
    q->handicap -= 4;

  } else if (q->handicap) {

    perf_score *= 2;
    --q->handicap;

  }

  /* Final adjustment based on input depth, under the assumption that fuzzing
     deeper test cases is more likely to reveal stuff that can't be
     discovered with traditional fuzzers. */

  switch (q->depth) {

    case 0 ... 3:
      break;
    case 4 ... 7:
      perf_score *= 2;
      break;
    case 8 ... 13:
      perf_score *= 3;
      break;
    case 14 ... 25:
      perf_score *= 4;
      break;
    default:
      perf_score *= 5;

  }

  u32         n_items;
  double      factor = 1.0;
  long double fuzz_mu;

  switch (afl->schedule) {

    case EXPLORE:
      break;

    case SEEK:
      break;

    case EXPLOIT:
      factor = MAX_FACTOR;
      break;

    case COE:
      fuzz_mu = 0.0;
      n_items = 0;

      // Don't modify perf_score for unfuzzed seeds
      if (!q->fuzz_level) break;

      u32 i;
      for (i = 0; i < afl->queued_items; i++) {

        if (likely(!afl->queue_buf[i]->disabled)) {

          fuzz_mu += log2(afl->n_fuzz[afl->queue_buf[i]->n_fuzz_entry]);
          n_items++;

        }

      }

      if (unlikely(!n_items)) { FATAL("Queue state corrupt"); }

      fuzz_mu = fuzz_mu / n_items;

      if (log2(afl->n_fuzz[q->n_fuzz_entry]) > fuzz_mu) {

        /* Never skip favourites */
        if (!q->favored) factor = 0;

        break;

      }

    // Fall through
    case FAST:

      // Don't modify unfuzzed seeds
      if (!q->fuzz_level) break;

      switch ((u32)log2(afl->n_fuzz[q->n_fuzz_entry])) {

        case 0 ... 1:
          factor = 4;
          break;

        case 2 ... 3:
          factor = 3;
          break;

        case 4:
          factor = 2;
          break;

        case 5:
          break;

        case 6:
          if (!q->favored) factor = 0.8;
          break;

        case 7:
          if (!q->favored) factor = 0.6;
          break;

        default:
          if (!q->favored) factor = 0.4;
          break;

      }

      if (q->favored) factor *= 1.15;

      break;

    case LIN:
      // Don't modify perf_score for unfuzzed seeds
      if (!q->fuzz_level) break;

      factor = q->fuzz_level / (afl->n_fuzz[q->n_fuzz_entry] + 1);
      break;

    case QUAD:
      // Don't modify perf_score for unfuzzed seeds
      if (!q->fuzz_level) break;

      factor =
          q->fuzz_level * q->fuzz_level / (afl->n_fuzz[q->n_fuzz_entry] + 1);
      break;

    case MMOPT:
      /* -- this was a more complex setup, which is good, but competed with
         -- rare. the simpler algo however is good when rare is not.
        // the newer the entry, the higher the pref_score
        perf_score *= (1 + (double)((double)q->depth /
        (double)afl->queued_items));
        // with special focus on the last 8 entries
        if (afl->max_depth - q->depth < 8) perf_score *= (1 + ((8 -
        (afl->max_depth - q->depth)) / 5));
      */
      // put focus on the last 5 entries
      if (afl->max_depth - q->depth < 5) { perf_score *= 2; }

      break;

    case RARE:

      // increase the score for every bitmap byte for which this entry
      // is the top contender
      perf_score += (q->tc_ref * 10);
      // the more often fuzz result paths are equal to this queue entry,
      // reduce its value
      perf_score *= (1 - (double)((double)afl->n_fuzz[q->n_fuzz_entry] /
                                  (double)afl->fsrv.total_execs));

      break;

    default:
      PFATAL("Unknown Power Schedule");

  }

  if (unlikely(afl->schedule >= EXPLOIT && afl->schedule <= QUAD)) {

    if (factor > MAX_FACTOR) { factor = MAX_FACTOR; }
    perf_score *= factor / POWER_BETA;

  }

  // MOpt mode
  if (afl->limit_time_sig != 0 && afl->max_depth - q->depth < 3) {

    perf_score *= 2;

  } else if (afl->schedule != COE && perf_score < 1) {

    // Add a lower bound to AFLFast's energy assignment strategies
    perf_score = 1;

  }

  /* Make sure that we don't go over limit. */

  if (perf_score > afl->havoc_max_mult * 100) {

    perf_score = afl->havoc_max_mult * 100;

  }

  return perf_score;

}

/* after a custom trim we need to reload the testcase from disk */

inline void queue_testcase_retake(afl_state_t *afl, struct queue_entry *q,
                                  u32 old_len) {

  if (likely(q->testcase_buf)) {

    u32 len = q->len;

    if (len != old_len) {

      afl->q_testcase_cache_size = afl->q_testcase_cache_size + len - old_len;
      q->testcase_buf = (u8 *)realloc(q->testcase_buf, len);

      if (unlikely(!q->testcase_buf)) {

        PFATAL("Unable to malloc '%s' with len %u", (char *)q->fname, len);

      }

    }

    int fd = open((char *)q->fname, O_RDONLY);

    if (unlikely(fd < 0)) { PFATAL("Unable to open '%s'", (char *)q->fname); }

    ck_read(fd, q->testcase_buf, len, q->fname);
    close(fd);

  }

}

/* after a normal trim we need to replace the testcase with the new data */

inline void queue_testcase_retake_mem(afl_state_t *afl, struct queue_entry *q,
                                      u8 *in, u32 len, u32 old_len) {

  if (likely(q->testcase_buf)) {

    u32 is_same = in == q->testcase_buf;

    if (likely(len != old_len)) {

      u8 *ptr = (u8 *)realloc(q->testcase_buf, len);

      if (likely(ptr)) {

        q->testcase_buf = ptr;
        afl->q_testcase_cache_size = afl->q_testcase_cache_size + len - old_len;

      }

    }

    if (unlikely(!is_same)) { memcpy(q->testcase_buf, in, len); }

  }

}

/* Returns the testcase buf from the file behind this queue entry.
  Increases the refcount. */

inline u8 *queue_testcase_get(afl_state_t *afl, struct queue_entry *q) {

  u32 len = q->len;

  /* first handle if no testcase cache is configured */

  if (unlikely(!afl->q_testcase_max_cache_size)) {

    u8 *buf;

    if (unlikely(q == afl->queue_cur)) {

      buf = (u8 *)afl_realloc((void **)&afl->testcase_buf, len);

    } else {

      buf = (u8 *)afl_realloc((void **)&afl->splicecase_buf, len);

    }

    if (unlikely(!buf)) {

      PFATAL("Unable to malloc '%s' with len %u", (char *)q->fname, len);

    }

    int fd = open((char *)q->fname, O_RDONLY);

    if (unlikely(fd < 0)) { PFATAL("Unable to open '%s'", (char *)q->fname); }

    ck_read(fd, buf, len, q->fname);
    close(fd);
    return buf;

  }

  /* now handle the testcase cache */

  if (unlikely(!q->testcase_buf)) {

    /* Buf not cached, let's load it */
    u32        tid = afl->q_testcase_max_cache_count;
    static u32 do_once = 0;  // because even threaded we would want this. WIP

    while (unlikely(
        afl->q_testcase_cache_size + len >= afl->q_testcase_max_cache_size ||
        afl->q_testcase_cache_count >= afl->q_testcase_max_cache_entries - 1)) {

      /* We want a max number of entries to the cache that we learn.
         Very simple: once the cache is filled by size - that is the max. */

      if (unlikely(afl->q_testcase_cache_size + len >=
                       afl->q_testcase_max_cache_size &&
                   (afl->q_testcase_cache_count <
                        afl->q_testcase_max_cache_entries &&
                    afl->q_testcase_max_cache_count <
                        afl->q_testcase_max_cache_entries) &&
                   !do_once)) {

        if (afl->q_testcase_max_cache_count > afl->q_testcase_cache_count) {

          afl->q_testcase_max_cache_entries =
              afl->q_testcase_max_cache_count + 1;

        } else {

          afl->q_testcase_max_cache_entries = afl->q_testcase_cache_count + 1;

        }

        do_once = 1;
        // release unneeded memory
        afl->q_testcase_cache = (struct queue_entry **)ck_realloc(
            afl->q_testcase_cache,
            (afl->q_testcase_max_cache_entries + 1) * sizeof(size_t));

      }

      /* Cache full. We neet to evict one or more to map one.
         Get a random one which is not in use */

      do {

        // if the cache (MB) is not enough for the queue then this gets
        // undesirable because q_testcase_max_cache_count grows sometimes
        // although the number of items in the cache will not change hence
        // more and more loops
        tid = rand_below(afl, afl->q_testcase_max_cache_count);

      } while (afl->q_testcase_cache[tid] == NULL ||

               afl->q_testcase_cache[tid] == afl->queue_cur);

      struct queue_entry *old_cached = afl->q_testcase_cache[tid];
      free(old_cached->testcase_buf);
      old_cached->testcase_buf = NULL;
      afl->q_testcase_cache_size -= old_cached->len;
      afl->q_testcase_cache[tid] = NULL;
      --afl->q_testcase_cache_count;
      ++afl->q_testcase_evictions;
      if (tid < afl->q_testcase_smallest_free)
        afl->q_testcase_smallest_free = tid;

    }

    if (unlikely(tid >= afl->q_testcase_max_cache_entries)) {

      // uh we were full, so now we have to search from start
      tid = afl->q_testcase_smallest_free;

    }

    // we need this while loop in case there were ever previous evictions but
    // not in this call.
    while (unlikely(afl->q_testcase_cache[tid] != NULL))
      ++tid;

    /* Map the test case into memory. */

    int fd = open((char *)q->fname, O_RDONLY);

    if (unlikely(fd < 0)) { PFATAL("Unable to open '%s'", (char *)q->fname); }

    q->testcase_buf = (u8 *)malloc(len);

    if (unlikely(!q->testcase_buf)) {

      PFATAL("Unable to malloc '%s' with len %u", (char *)q->fname, len);

    }

    ck_read(fd, q->testcase_buf, len, q->fname);
    close(fd);

    /* Register testcase as cached */
    afl->q_testcase_cache[tid] = q;
    afl->q_testcase_cache_size += len;
    ++afl->q_testcase_cache_count;
    if (likely(tid >= afl->q_testcase_max_cache_count)) {

      afl->q_testcase_max_cache_count = tid + 1;

    } else if (unlikely(tid == afl->q_testcase_smallest_free)) {

      afl->q_testcase_smallest_free = tid + 1;

    }

  }

  return q->testcase_buf;

}

/* Adds the new queue entry to the cache. */

inline void queue_testcase_store_mem(afl_state_t *afl, struct queue_entry *q,
                                     u8 *mem) {

  u32 len = q->len;

  if (unlikely(afl->q_testcase_cache_size + len >=
                   afl->q_testcase_max_cache_size ||
               afl->q_testcase_cache_count >=
                   afl->q_testcase_max_cache_entries - 1)) {

    // no space? will be loaded regularly later.
    return;

  }

  u32 tid;

  if (unlikely(afl->q_testcase_max_cache_count >=
               afl->q_testcase_max_cache_entries)) {

    // uh we were full, so now we have to search from start
    tid = afl->q_testcase_smallest_free;

  } else {

    tid = afl->q_testcase_max_cache_count;

  }

  while (unlikely(afl->q_testcase_cache[tid] != NULL))
    ++tid;

  /* Map the test case into memory. */

  q->testcase_buf = (u8 *)malloc(len);

  if (unlikely(!q->testcase_buf)) {

    PFATAL("Unable to malloc '%s' with len %u", (char *)q->fname, len);

  }

  memcpy(q->testcase_buf, mem, len);

  /* Register testcase as cached */
  afl->q_testcase_cache[tid] = q;
  afl->q_testcase_cache_size += len;
  ++afl->q_testcase_cache_count;

  if (likely(tid >= afl->q_testcase_max_cache_count)) {

    afl->q_testcase_max_cache_count = tid + 1;

  } else if (unlikely(tid == afl->q_testcase_smallest_free)) {

    afl->q_testcase_smallest_free = tid + 1;

  }

}

