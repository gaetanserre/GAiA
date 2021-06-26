#ifndef DOT_H
#define DOT_H

#include <vector>
#include <iostream>


struct thread_data {
  const std::vector<double>* u;
  const std::vector<double>* v;
  const int* lo;
  const int* hi;
  double* res;
};

static void* dot_thread (void *threadarg) {
  struct thread_data *td;
  td = (struct thread_data *) threadarg;
  *(*td).res = 0;
  for (int i = *(*td).lo; i<*(*td).hi; i++) {
    *(*td).res += (*(*td).u)[i] * (*(*td).v)[i];
  }
  return NULL;
}

static double dot(const std::vector<double>& u, const std::vector<double>& v, int nb_thread) {
  int u_size = u.size(); int v_size = v.size();
  if (u_size != v_size) {
    std::string error = "Dot product between vector of size " + std::to_string(u_size) + " and " + "vector of size " + std::to_string(v_size) + ".";
    throw std::runtime_error(error);
  } else {

    if (nb_thread == 1) {
      int lo = 0; double res = 0;
      thread_data t {&u, &v, &lo, &u_size, &res};
      dot_thread(&t);
      return res;
    }

    double res = 0;
    pthread_t threads[nb_thread];
    thread_data tds[nb_thread];
    int los[nb_thread];
    int his[nb_thread];
    double temps[nb_thread];
    int step = u_size / nb_thread;
    for (int i = 0; i<nb_thread; i++) {
      los[i] = i * step;
      his[i] = los[i] + step;
      if (i == nb_thread - 1 && his[i] < u_size) his[i] = u_size;
      tds[i] = {&u, &v, &los[i], &his[i], &temps[i]};
      pthread_create(&threads[i], NULL, dot_thread, (void *)&tds[i]);
    }
    for (int i = 0; i<nb_thread; i++) {
      pthread_join(threads[i], NULL);
      res += *tds[i].res;
    }
    return res;
  }
}
#endif // DOT_H