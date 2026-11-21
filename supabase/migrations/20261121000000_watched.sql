-- Watch history imported from Netflix, IMDb or Letterboxd (backend/app/db.py WatchedItem): implicit
-- feedback, counted as liked unless the user rates the film lower. One row per film; times counts rewatches.
create table if not exists public.watched (
    user_id     uuid not null references auth.users (id) on delete cascade,
    movie_id    integer not null,
    watched_at  timestamptz not null,
    source      varchar(20) not null check (source in ('netflix', 'imdb', 'letterboxd')),
    times       integer not null default 1 check (times >= 1),
    imported_at timestamptz not null default now(),
    primary key (user_id, movie_id)
);

alter table public.watched enable row level security;
create policy "own watch history" on public.watched for all using (auth.uid() = user_id) with check (auth.uid() = user_id);
