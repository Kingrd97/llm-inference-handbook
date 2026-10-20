#!/usr/bin/env python3
"""CPU teaching lab: causal attention, paged KV, prefix reuse, continuous scheduling.

No trained model, GPU, async work, stochastic sampling, or timing simulation.
Run: python3 inference_lab.py --test | --demo
Cache owns references here; real frameworks can use different accounting.
"""
from __future__ import annotations

import argparse
import math
import random
import unittest
from collections import Counter, OrderedDict
from dataclasses import dataclass, field

KV = tuple[tuple[float, ...], tuple[float, ...]]


def matvec(matrix, vector):
    return [sum(a * b for a, b in zip(row, vector)) for row in matrix]


def softmax(values):
    weights = [math.exp(v - max(values)) for v in values]
    return [v / sum(weights) for v in weights]


class TinyModel:
    """One untrained causal attention layer. NOT a pretrained Transformer."""
    def __init__(self, seed=7, dim=8, vocab=11):
        self.dim, self.vocab = dim, vocab
        rng = random.Random(seed)
        def matrix(rows, cols):
            return [[rng.uniform(-0.5, 0.5) for _ in range(cols)] for _ in range(rows)]
        self.embedding = matrix(vocab, dim)
        self.wq, self.wk, self.wv, self.wo = [matrix(dim, dim) for _ in range(4)]
        self.head = matrix(vocab, dim)

    def input_vector(self, token, position):
        return [v + 0.1 * math.sin(position / (j + 1))
                for j, v in enumerate(self.embedding[token])]

    def project(self, token, position):
        x = self.input_vector(token, position)
        return x, matvec(self.wq, x), (tuple(matvec(self.wk, x)), tuple(matvec(self.wv, x)))

    def finish(self, x, q, kvs):
        scores = [sum(a * b for a, b in zip(q, k)) / math.sqrt(self.dim) for k, _ in kvs]
        weights = softmax(scores)
        context = [sum(w * kv[1][j] for w, kv in zip(weights, kvs)) for j in range(self.dim)]
        mixed = matvec(self.wo, context)
        hidden = [math.tanh(a + b) for a, b in zip(x, mixed)]
        return matvec(self.head, hidden)

    def full_logits(self, tokens):
        # Reference: re-project every token, with no cached state or page table.
        projected = [self.project(t, i) for i, t in enumerate(tokens)]
        x, q, _ = projected[-1]
        return self.finish(x, q, [item[2] for item in projected])

    def step(self, token, position, history):
        assert position == len(history)
        x, q, kv = self.project(token, position)
        return self.finish(x, q, history + [kv]), kv


class PagePool:
    def __init__(self, count=32, page_size=4):
        assert count > 0 and page_size > 0
        self.page_size = page_size
        self.data: list[list[KV]] = [[] for _ in range(count)]
        self.refs = [0] * count
        self.free = list(reversed(range(count)))

    def acquire(self):
        if not self.free:
            raise MemoryError('No free KV page')
        page = self.free.pop()
        assert self.refs[page] == 0
        self.refs[page] = 1
        self.data[page] = []
        return page

    def retain(self, page):
        assert self.refs[page] > 0
        self.refs[page] += 1

    def release(self, page):
        assert self.refs[page] > 0, 'Double release'
        self.refs[page] -= 1
        if self.refs[page] == 0:
            self.data[page] = []
            self.free.append(page)

    def read(self, pages, count):
        result = [kv for page in pages for kv in self.data[page]]
        assert len(result) == count
        return result

    def append(self, pages, position, kv):
        if position % self.page_size == 0:
            pages.append(self.acquire())
        page = pages[-1]
        assert self.refs[page] == 1, 'Shared page must be immutable'
        assert len(self.data[page]) == position % self.page_size
        self.data[page].append(kv)

    def check(self, owners):
        counts = Counter(owners)
        assert len(self.free) == len(set(self.free))
        assert set(self.free) == {p for p, n in enumerate(self.refs) if n == 0}
        assert self.refs == [counts[p] for p in range(len(self.refs))]
        assert all(len(data) <= self.page_size for data in self.data)


class PrefixCache:
    """Full-page keys contain the entire prefix; inefficient but transparent.

    One fixed model per Engine is the namespace. Only completed prompt pages
    are published. No partial-page sharing, so no copy-on-write is needed.
    """
    def __init__(self, pool):
        self.pool = pool
        self.entries = OrderedDict()

    def publish(self, prompt, pages):
        size = self.pool.page_size
        for i, page in enumerate(pages):
            end = (i + 1) * size
            if end > len(prompt):
                break
            key = tuple(prompt[:end])
            assert len(self.pool.data[page]) == size
            if key not in self.entries:
                self.pool.retain(page)
                self.entries[key] = page

    def attach(self, prompt):
        pages = []
        size = self.pool.page_size
        # Keep at least one prompt token for producing the first output logits.
        for end in range(size, len(prompt), size):
            key = tuple(prompt[:end])
            if key not in self.entries:
                break
            page = self.entries[key]
            self.pool.retain(page)
            pages.append(page)
            self.entries.move_to_end(key)
        return pages, len(pages) * size

    def clear(self):
        for page in self.entries.values():
            self.pool.release(page)
        self.entries.clear()


@dataclass
class Request:
    name: str
    prompt: list[int]
    max_new: int
    pages: list[int] = field(default_factory=list)
    output: list[int] = field(default_factory=list)
    computed: int = 0
    hit_tokens: int = 0
    executed_tokens: int = 0
    state: str = 'waiting'
    first_tick: int | None = None
    end_tick: int | None = None

    @property
    def known(self):
        return self.prompt + self.output


class Engine:
    def __init__(self, page_count=32, page_size=4, token_budget=4, prefill_chunk=2):
        assert token_budget > 0 and prefill_chunk > 0
        self.model = TinyModel()
        self.pool = PagePool(page_count, page_size)
        self.cache = PrefixCache(self.pool)
        self.requests: dict[str, Request] = {}
        self.budget, self.chunk = token_budget, prefill_chunk
        self.round = 0

    def add(self, name, prompt, max_new=4):
        if name in self.requests:
            raise ValueError('Duplicate request id')
        if not prompt or max_new <= 0 or any(not 0 <= t < self.model.vocab for t in prompt):
            raise ValueError('Need nonempty valid prompt and positive max_new')
        # No preemption in this teaching engine. Reject a request that cannot
        # fit even when alone. Last emitted token need not have computed KV.
        if len(prompt) + max_new - 1 > len(self.pool.refs) * self.pool.page_size:
            raise ValueError('Request exceeds total KV capacity')
        req = Request(name, list(prompt), max_new)
        req.pages, req.computed = self.cache.attach(req.prompt)
        req.hit_tokens = req.computed
        self.requests[name] = req
        self.check()
        return req

    def finish(self, req, state):
        if req.state in ('finished', 'cancelled'):
            return
        req.state, req.end_tick = state, self.round
        for page in req.pages:
            self.pool.release(page)
        req.pages.clear()

    def cancel(self, name):
        self.finish(self.requests[name], 'cancelled')
        self.check()

    def active(self):
        return [r for r in self.requests.values() if r.state not in ('finished', 'cancelled')]

    def tick(self):
        self.round += 1
        budget, actions = self.budget, []
        # Existing decode first; stable arrival order within each group.
        candidates = sorted(self.active(), key=lambda r: r.computed < len(r.prompt))
        for req in candidates:
            is_prefill = req.computed < len(req.prompt)
            count = min(budget, self.chunk if is_prefill else 1,
                        len(req.prompt) - req.computed if is_prefill else 1)
            used = 0
            for _ in range(count):
                if req.computed % self.pool.page_size == 0 and not self.pool.free:
                    # Deliberately simple eviction: drop all cache ownership,
                    # leaving active requests' references intact.
                    self.cache.clear()
                    if not self.pool.free:
                        break
                position = req.computed
                logits, kv = self.model.step(req.known[position], position,
                                            self.pool.read(req.pages, position))
                self.pool.append(req.pages, position, kv)
                req.computed += 1
                req.executed_tokens += 1
                req.state = 'running'
                used += 1
                budget -= 1
                if req.computed == len(req.prompt) and not req.output:
                    self.cache.publish(req.prompt, req.pages)
                if req.computed == len(req.known):
                    token = max(range(self.model.vocab), key=logits.__getitem__)
                    req.output.append(token)
                    if req.first_tick is None:
                        req.first_tick = self.round
                    if len(req.output) == req.max_new:
                        self.finish(req, 'finished')
                        break
            if used:
                actions.append(f'{req.name}:{"P" if is_prefill else "D"}{used}')
            if budget == 0:
                break
        self.check()
        if self.active() and not actions:
            raise RuntimeError('No progress: active KV exhausted; cancel or use more pages')
        return actions

    def run(self, limit=1000):
        for _ in range(limit):
            if not self.active():
                return
            self.tick()
        raise RuntimeError('Iteration limit')

    def check(self):
        owners = list(self.cache.entries.values())
        for req in self.active():
            owners.extend(req.pages)
            assert len(self.pool.read(req.pages, req.computed)) == req.computed
            assert req.computed <= len(req.known)
        self.pool.check(owners)

    def reference(self, prompt, max_new):
        tokens = list(prompt)
        for _ in range(max_new):
            logits = self.model.full_logits(tokens)
            tokens.append(max(range(self.model.vocab), key=logits.__getitem__))
        return tokens[len(prompt):]


class LabTests(unittest.TestCase):
    def test_incremental_logits_against_full_reference(self):
        model = TinyModel()
        rng = random.Random(31)
        for length in (1, 3, 4, 5, 17, 33):
            tokens = [rng.randrange(model.vocab) for _ in range(length)]
            history = []
            for i, token in enumerate(tokens):
                logits, kv = model.step(token, i, history)
                ref = model.full_logits(tokens[:i + 1])
                self.assertLess(max(abs(a-b) for a, b in zip(logits, ref)), 1e-12)
                history.append(kv)

    def test_batch_chunk_page_variants_match_reference(self):
        for page_size in (1, 2, 4, 7):
            for budget in (1, 3, 8):
                engine = Engine(page_size=page_size, token_budget=budget)
                for i, prompt in enumerate(([1, 2, 3], [3] * 9, [8, 2, 7, 4, 1])):
                    engine.add(str(i), list(prompt), 5)
                engine.run()
                for req in engine.requests.values():
                    self.assertEqual(req.output, engine.reference(req.prompt, 5))
                    self.assertEqual(req.executed_tokens, len(req.prompt) + 4)
                engine.cache.clear()
                engine.check()
                self.assertEqual(len(engine.pool.free), len(engine.pool.refs))

    def test_shared_prefix_and_private_tail(self):
        engine = Engine()
        a = engine.add('A', [1, 2, 3, 4, 5, 6], 8)
        while a.first_tick is None:
            engine.tick()
        b = engine.add('B', [1, 2, 3, 4, 9], 4)
        self.assertEqual(b.hit_tokens, 4)
        self.assertEqual(a.pages[0], b.pages[0])
        self.assertEqual(engine.pool.refs[a.pages[0]], 3)  # A, B, cache
        engine.run()
        self.assertEqual(b.output, engine.reference(b.prompt, b.max_new))
        self.assertEqual(b.executed_tokens, len(b.prompt) + b.max_new - 1 - 4)
        engine.cache.clear()
        engine.check()
        self.assertEqual(len(engine.pool.free), 32)

    def test_cancel_prefill_decode_and_idempotence(self):
        engine = Engine(token_budget=2, prefill_chunk=2)
        a = engine.add('A', [1] * 9, 3)
        engine.tick()
        engine.cancel('A')
        engine.cancel('A')
        self.assertFalse(a.output)
        b = engine.add('B', [2, 3, 4, 5, 6], 8)
        while b.first_tick is None:
            engine.tick()
        old = list(b.output)
        engine.cancel('B')
        engine.tick()
        self.assertEqual(b.output, old)
        engine.cache.clear()
        engine.check()
        self.assertEqual(len(engine.pool.free), 32)

    def test_allocation_failure_atomic_and_double_release(self):
        pool = PagePool(1, 2)
        page = pool.acquire()
        with self.assertRaises(MemoryError):
            pool.acquire()
        pool.check([page])
        pool.release(page)
        with self.assertRaises(AssertionError):
            pool.release(page)
        pool.check([])

    def test_capacity_stall_reported_then_cancel_recovers(self):
        engine = Engine(page_count=2, page_size=2, token_budget=2, prefill_chunk=1)
        engine.add('A', [1, 2, 3], 2)
        engine.add('B', [4, 5, 6], 2)
        engine.tick()
        engine.tick()
        with self.assertRaisesRegex(RuntimeError, 'No progress'):
            engine.tick()
        engine.cancel('B')
        engine.run()
        self.assertEqual(engine.requests['A'].output, engine.reference([1, 2, 3], 2))
        engine.cache.clear()
        engine.check()
        self.assertEqual(len(engine.pool.free), 2)

    def test_eviction_reclaims_idle_cache(self):
        engine = Engine(page_count=2, page_size=2)
        engine.add('A', [1, 2, 3], 1)
        engine.run()
        self.assertTrue(engine.cache.entries)
        engine.add('B', [4, 5, 6], 2)
        engine.run()
        self.assertEqual(engine.requests['B'].output, engine.reference([4, 5, 6], 2))
        engine.cache.clear()
        engine.check()
        self.assertEqual(len(engine.pool.free), 2)

    def test_reject_oversized_and_invalid_requests(self):
        engine = Engine(page_count=2, page_size=2)
        with self.assertRaises(ValueError):
            engine.add('A', [1] * 5, 1)
        with self.assertRaises(ValueError):
            engine.add('B', [], 1)
        with self.assertRaises(ValueError):
            engine.add('C', [999], 1)
        engine.check()


def demo():
    engine = Engine(token_budget=4, prefill_chunk=2)
    engine.add('A', [1, 2, 3, 4, 5, 6], 6)
    print('CPU toy model; tick is NOT wall-clock time. P=prefill, D=decode.')
    added = False
    while engine.active():
        if engine.requests['A'].first_tick is not None and not added:
            b = engine.add('B', [1, 2, 3, 4, 9], 4)
            engine.add('C', [7] * 9, 3)
            print(f'  ARRIVE B: prefix_hit={b.hit_tokens}; ARRIVE C')
            added = True
        if engine.round == 5 and 'C' in engine.requests:
            engine.cancel('C')
            print('  CANCEL C')
        actions = engine.tick()
        print(f'tick={engine.round:2} batch={actions} free_pages={len(engine.pool.free):2}')
        for req in engine.requests.values():
            print(f'  {req.name}: {req.state:9} computed={req.computed:2} '
                  f'output={req.output} physical_pages={req.pages}')
    for req in engine.requests.values():
        if req.state == 'finished':
            assert req.output == engine.reference(req.prompt, req.max_new)
        print(f'{req.name}: hit={req.hit_tokens}, executed={req.executed_tokens}, '
              f'first_tick={req.first_tick}, end_tick={req.end_tick}')
    engine.cache.clear()
    engine.check()
    print(f'Cache cleared; all {len(engine.pool.free)} pages returned.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()
    if args.test:
        unittest.main(argv=['inference_lab.py'], verbosity=2, exit=True)
    else:
        demo()
