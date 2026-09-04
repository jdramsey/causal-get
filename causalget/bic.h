#ifndef BIC_H_
#define BIC_H_

// Floor on the residual variance in bic_update (see the comment there). 1e-12 corresponds to a
// partial R^2 of 1 - 1e-12, i.e. a numerically exact linear dependence.
#ifndef BIC_MIN_RESID_VAR
#define BIC_MIN_RESID_VAR 1e-12
#endif

#include <stdlib.h>
#include <math.h>

#define TNU(x) (x * (x + 1) >> 1)
#define GET(X, r, c) X[TNU(r) + c]

#define BTA_IMPLEMENTATION

#ifndef BTA_H_
#include "bta.h"
#endif // BTA_H_

typedef struct BIC BIC;

struct BIC {
  // parameters
  float discount;
  double tol;      // min improvement, in BIC points, for better_mutation to accept a move
  double lambda;   // singularity lambda (Tetrad's SemBicScore.setLambda): the regression coefficients are
                   // ridge, b = (Sigma_BB + lambda I)^-1 sigma_By, but the residual variance is the true
                   // residual variance of y - b'x under the UNregularized covariance, as in
                   // SemBicScore.getResidualVariance: bStar' Cov bStar. See bic_update.

  // data
  double *X;       // covariance (m x m, row-major) or data (m x n, variable-major), see get_cov
  size_t n;
  size_t p;        // number of search VARIABLES (what BOSS/GST permute and pick parents from)
  size_t m;        // number of embedded COLUMNS (== p for plain linear BIC)

  // Basis-function embedding: variable v owns columns offsets[v] .. offsets[v+1]-1 of the
  // covariance. NULL means the identity embedding (column v is variable v). With a non-trivial
  // embedding the local score is Tetrad's BasisFunctionBicScore: the chain-rule (joint Gaussian)
  // BIC of the child's block given the union of the parents' blocks.
  uint32_t *offsets;

  // covariance (indexed by COLUMN, not variable)
  double (*get_cov) (BIC *bic, uint32_t a, uint32_t b);

  // LDL, one row per embedded column of the current parent set (in z order), followed by the
  // rows of the child block. Rows are addressed by column position, not variable position.
  double *L;
  double *D;       // 1/sqrt(regularized residual variance) per row, used to factor later rows
  double *res;     // residual variance per row (Tetrad's definition), what the score is computed from
  double *b;       // scratch (m) for the back-substitution when lambda > 0
  uint32_t *cols;  // embedded column id of each row
  size_t q;        // number of parent VARIABLES currently in z

  // indices
  uint32_t y;
  uint32_t *z;
};

// Block bounds of variable v.
static inline uint32_t bic_col_lo(const BIC *bic, uint32_t v) { return bic->offsets ? bic->offsets[v] : v; }
static inline uint32_t bic_col_hi(const BIC *bic, uint32_t v) { return bic->offsets ? bic->offsets[v + 1] : v + 1; }

// LDL row at which the block of parent slot i (0 <= i <= q) begins, i.e. the number of embedded
// columns contributed by z[0..i). O(i); i is a parent-set size so this is negligible next to the
// O(rows^2) factor update itself.
static inline size_t bic_row_start(const BIC *bic, size_t i)
{
  if (!bic->offsets) return i;
  size_t r = 0;
  for (size_t j = 0; j < i; j++) r += bic->offsets[bic->z[j] + 1] - bic->offsets[bic->z[j]];
  return r;
}

// Total embedded columns in the current parent set z[0..q).
static inline size_t bic_parent_cols(const BIC *bic) { return bic_row_start(bic, bic->q); }


double get_cov_precomp(BIC *bic, uint32_t a, uint32_t b);
double get_cov_onfly(BIC *bic, uint32_t a, uint32_t b);

void bic_update(BIC *bic, uint32_t x);
double bic_score(BIC *bic);

int bic_contains(uint32_t *z, size_t size, uint32_t x);
int bic_find(uint32_t *z, size_t size, uint32_t x);

void bic_grow(BIC *bic, Bit_Array prefix);
void bic_shrink(BIC *bic);

#endif // BIC_H_

#ifdef BIC_IMPLEMENTATION


double get_cov_precomp(BIC *bic, uint32_t a, uint32_t b)
{
  return bic->X[a * bic->m + b];
}


double get_cov_onfly(BIC *bic, uint32_t a, uint32_t b)
{
  size_t n = bic->n;
  if (n < 2) return 0.0;

  double *x = &bic->X[a * n];
  double *y = &bic->X[b * n];

  double mu_x = 0.0;
  double mu_y = 0.0;
  for (size_t i = 0; i < n; i++) {
    mu_x += x[i];
    mu_y += y[i];
  }
  mu_x /= n;
  mu_y /= n;

  double cov_xy = 0.0;
  for (size_t i = 0; i < n; i++) {
    cov_xy += (x[i] - mu_x) * (y[i] - mu_y);
  }
  cov_xy /= n;

  return cov_xy;
}


// Appends the block of variable x as the next rows of the LDL, given the parents z[0..q).
// Each column of the block is factored against every earlier row, including the block's own
// earlier columns -- that within-block conditioning is what makes the summed residual
// log-variances telescope to log det of the conditional covariance of the block, i.e. a joint
// Gaussian likelihood (and hence a score-equivalent BIC). With the identity embedding this is
// exactly the original one-row update.
void bic_update(BIC *bic, uint32_t x)
{
  double *L = bic->L;
  double *D = bic->D;
  uint32_t *cols = bic->cols;
  size_t row = bic_row_start(bic, bic->q);

  for (uint32_t c = bic_col_lo(bic, x); c < bic_col_hi(bic, x); c++, row++) {
    for (size_t j = 0; j < row; j++) {
      double acc = bic->get_cov(bic, c, cols[j]);
      for (size_t k = 0; k < j; k++) {
        acc -= GET(L, row, k) * GET(L, j, k);
      }
      GET(L, row, j) = acc * D[j];
    }

    // Residual variance of column c given all earlier rows (on standardized data, in (0, 1]).
    // Accumulated in the same order as the original single-row code so that the linear
    // (identity-embedding) path is bit-identical to it.
    double res = bic->get_cov(bic, c, c);
    for (size_t k = 0; k < row; k++) {
      res -= GET(L, row, k) * GET(L, row, k);
    }

    // Tetrad's ridge residual. With L the Cholesky factor of (Sigma_BB + lambda I) and w = L^-1 sigma
    // (the L entries of this row), the ridge coefficients are b = L^-T w and Tetrad's residual
    //   sigma_yy - 2 b'sigma + b'Sigma_BB b  =  sigma_yy - w'w - lambda ||b||^2,
    // since b'Sigma_BB b = b'(Sigma_BB + lambda I) b - lambda ||b||^2 = w'w - lambda ||b||^2.
    // The res computed above is sigma_yy - w'w; one back-substitution gives the correction.
    if (bic->lambda > 0.0 && row > 0) {
      double *b = bic->b;
      double bb = 0.0;
      for (size_t k = row; k-- > 0;) {
        double acc = GET(L, row, k);
        for (size_t j = k + 1; j < row; j++) acc -= GET(L, j, k) * b[j];
        b[k] = acc * D[k];
        bb += b[k] * b[k];
      }
      res -= bic->lambda * bb;
    }

    // A numerically singular correlation matrix -- exploding or collinear series, more variables than
    // rows -- drives it to zero or slightly negative, and 1/sqrt then gives inf or NaN. NaN scores never
    // compare as improvements, so better_mutation's do/while can fail to terminate. Clamp at a floor that
    // is far below anything a real residual reaches; the score stays finite and the search converges.
    if (!(res > BIC_MIN_RESID_VAR)) res = BIC_MIN_RESID_VAR;
    bic->res[row] = res;

    // The regularized pivot is what later rows are factored against: adding lambda here is the same
    // as inverting (Sigma_BB + lambda I) in every regression that has this column as a regressor.
    // With lambda == 0 this is exactly the original D[i]. (The pivot needs sigma_yy - w'w + lambda,
    // not the Tetrad residual above, which is why it is recomputed from the L row.)
    double dreg = bic->get_cov(bic, c, c) + bic->lambda;
    for (size_t k = 0; k < row; k++) dreg -= GET(L, row, k) * GET(L, row, k);
    if (!(dreg > BIC_MIN_RESID_VAR)) dreg = BIC_MIN_RESID_VAR;
    D[row] = 1.0 / sqrt(dreg);
    cols[row] = c;
  }
}


// Local score of y given z[0..q), in units of (Tetrad BIC) / (2n) up to an additive constant
// that depends only on the block sizes: Tetrad's 2*sum(lik) - c*dof*log(n) with
// lik = -n/2 (log(2 pi sigma^2) + 1) and dof = number of regressors, summed over the child's
// columns in chain-rule order (column t is regressed on the parents' columns plus the t earlier
// columns of its own block). Constant dropped: -n/2 (log 2 pi + 1) per child column.
double bic_score(BIC *bic)
{
  bic_update(bic, bic->y);

  double c = bic->discount;                            // discount
  size_t b = bic_parent_cols(bic);                     // parent columns
  size_t k = bic_col_hi(bic, bic->y) - bic_col_lo(bic, bic->y); // child columns
  double logn = log((double)bic->n) / (2.0 * bic->n);  // logn / 2n

  // log(1/sqrt(res)) rather than -0.5*log(res): same value, but the former is what the original
  // code computed (via D), and keeping it bit-identical keeps the linear path's tie-breaking identical.
  double ll = 0.0;
  for (size_t t = 0; t < k; t++) ll += log(1.0 / sqrt(bic->res[b + t]));
  size_t dof = k * b + (k * (k - 1)) / 2;

  return ll - c * (double)dof * logn;
}


// change to bool?
int bic_contains(uint32_t *z, size_t size, uint32_t x)
{
  for (size_t i = 0; i < size; i++) {
    if (z[i] == x) return 1;
  }

  return 0;
}


// change to uint32_t or size_t?
int bic_find(uint32_t *z, size_t size, uint32_t x)
{
  for (size_t i = 0; i < size; i++) {
    if (z[i] == x) return i;
  }

  // if not found return the last element 
  // (assumes the item is in the list)
  return size;
}

// change to skip
// void bic_grow(BIC *bic, Bit_Array skip)
void bic_grow(BIC *bic, Bit_Array prefix)
{
  size_t p = bic->p;
  uint32_t y = bic->y;
  uint32_t *z = bic->z;

  int add = -1;
  double score;
  double best = bic_score(bic);

  while(bic->q < p) {
    for (uint32_t x = 0; x < p; x++) {
      if (x == y) continue;
      // if (bta_check(skip, x)) continue; // change to skip?
      if (!bta_check(prefix, x)) continue; // is this correct?
      if (bic_contains(z, bic->q, x)) continue;

      bic_update(bic, x);
      z[bic->q++] = x;
      score = bic_score(bic);
      bic->q--;

      if (score > best) { 
        best = score;
        add = x;
      }
    }

    if (add != -1) {
      bic_update(bic, add);
      z[bic->q++] = add;
      add = -1;
    } else break;
  }
}


void bic_shrink(BIC *bic)
{
  if (bic->q == 0) return;
  uint32_t *z = bic->z;

  int del = -1;
  double score;
  double best = bic_score(bic);

  for (size_t size = bic->q - 1; size > 0; size--) {
    for (size_t i = 0; i < size + 1; i++) {

      uint32_t x = z[size - i];
      z[size - i] = z[size];
      z[size] = x;

      for (bic->q = size - i; bic->q < size; bic->q++) {
        bic_update(bic, z[bic->q]);
      }
      score = bic_score(bic);

      if (score > best) { 
        best = score;
        del = x;
      }
    }

    if (del != -1) {
      int i = bic_find(z, size, del);
      z[i] = z[size];
      for (bic->q = i; bic->q < size; bic->q++) {
        bic_update(bic, z[bic->q]);
      }
      del = -1;
    } else {
      bic_update(bic, z[bic->q]);
      bic->q++;
      break;
    }
  }
}

#endif // BIC_IMPLEMENTATION
