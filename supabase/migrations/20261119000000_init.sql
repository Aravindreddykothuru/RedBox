-- Red Box schema for Supabase Postgres.
-- Mirrors backend/app/db.py. The movie catalog is NOT stored here: it ships inside the model bundle,
-- so the catalog always matches the deployed models. The database only stores what users do.

create table if not exists public.profiles (
    user_id        uuid primary key references auth.users (id) on delete cascade,
    display_name   varchar(80),
    favorite_genres jsonb not null default '[]'::jsonb,
    onboarded_at   timestamptz,
    created_at     timestamptz not null default now()
);

create table if not exists public.ratings (
    user_id    uuid not null references auth.users (id) on delete cascade,
    movie_id   integer not null,
    rating     double precision not null check (rating >= 0.5 and rating <= 5.0),
    created_at timestamptz not null default now(),  -- order of the user's history (sequential models)
    updated_at timestamptz not null default now(),
    primary key (user_id, movie_id)
);
create index if not exists ratings_user_created_idx on public.ratings (user_id, created_at);

create table if not exists public.watchlist (
    user_id  uuid not null references auth.users (id) on delete cascade,
    movie_id integer not null,
    added_at timestamptz not null default now(),
    primary key (user_id, movie_id)
);

create table if not exists public.events (
    id           bigint generated always as identity primary key,
    user_id      uuid not null references auth.users (id) on delete cascade,
    movie_id     integer not null,
    type         varchar(20) not null check (type in ('impression', 'click', 'watchlist', 'rate')),
    source_model varchar(32),
    position     integer check (position >= 0),
    created_at   timestamptz not null default now()
);
create index if not exists events_user_idx on public.events (user_id);

-- Row-level security: the API checks the signed-in user on every request; RLS is the second line of
-- defence, so even a client talking to the database directly can only touch its own rows.
alter table public.profiles  enable row level security;
alter table public.ratings   enable row level security;
alter table public.watchlist enable row level security;
alter table public.events    enable row level security;

create policy "own profile"   on public.profiles  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);
create policy "own ratings"   on public.ratings   for all using (auth.uid() = user_id) with check (auth.uid() = user_id);
create policy "own watchlist" on public.watchlist for all using (auth.uid() = user_id) with check (auth.uid() = user_id);
-- events are append-only for users (analysis reads them with the service role)
create policy "insert own events" on public.events for insert with check (auth.uid() = user_id);
